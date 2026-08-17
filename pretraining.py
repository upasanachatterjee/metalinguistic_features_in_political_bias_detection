"""Multi-task pretraining entry point: one shared RoBERTa backbone, up to four
objectives (triplet, MLM, themes, tone). Run it as
`accelerate launch pretraining.py --config run_configs/tlp_tone_16.yaml`.
"""

import argparse
import datetime
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.state import PartialState
from accelerate.utils import set_seed
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import build_dataloader
from gradient_diagnostics import GradientDiagnostics
from model import MultiTaskRoberta
from config import RunConfig, load_run_config, training_progress
from training_log import TrainingLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

SEED = 42

# Number of recent step durations averaged for the run-level ETA.
ETA_WINDOW = 50

TASKS = ("triplet", "mlm", "tone", "themes")

# Logical task name -> the key MultiTaskRoberta.forward returns its raw loss under.
TASK_LOSS_KEYS = {
    "triplet": "triplet_loss",
    "themes": "theme_loss",
    "tone": "tone_loss",
    "mlm": "mlm_loss",
}

TASK_WEIGHTED_LOSS_KEYS = {
    task: key.replace("_loss", "_weighted_loss") for task, key in TASK_LOSS_KEYS.items()
}

# Collator key -> prefixed key: this mapping IS the collators/model.py contract.
TASK_BATCH_KEYS = {
    "triplet": {
        "a_ids": "triplet_a_ids",
        "a_mask": "triplet_a_mask",
        "p_ids": "triplet_p_ids",
        "p_mask": "triplet_p_mask",
        "n_ids": "triplet_n_ids",
        "n_mask": "triplet_n_mask",
    },
    "themes": {
        "input_ids": "theme_input_ids",
        "attention_mask": "theme_attention_mask",
        "labels": "theme_labels",
    },
    "tone": {
        "input_ids": "tone_input_ids",
        "attention_mask": "tone_attention_mask",
        "targets": "tone_labels",
    },
    "mlm": {
        "input_ids": "mlm_input_ids",
        "attention_mask": "mlm_attention_mask",
        "labels": "mlm_labels",
    },
}


def log(message: str = "") -> None:
    """Print on rank 0 only; the other ranks would just duplicate the line."""
    if PartialState().is_main_process:
        print(message, flush=True)


@dataclass
class PreparedRun:
    """What Accelerate handed back, plus the step counts derived from the data.

    A plain record passed to `train_one_epoch` so its signature stays short --
    it has no behaviour of its own.
    """

    accelerator: Accelerator
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    dataloader: Any
    # A task in cfg.tasks can be missing here (themes, with no label space).
    active_tasks: List[str]
    max_steps_per_epoch: int
    total_steps: int


@dataclass
class TrainingState:
    """Counters that persist across epochs."""
    step: int = 0
    micro_step: int = 0
    started_at: float = field(default_factory=time.time)
    # Recent step durations, for the run-level ETA.
    step_times: Deque[float] = field(default_factory=lambda: deque(maxlen=ETA_WINDOW))
    # Raw per-task losses, averaged and flushed every args.log_every steps.
    task_losses: Dict[str, List[float]] = field(
        default_factory=lambda: {task: [] for task in TASKS}
    )
    weighted_losses: Dict[str, List[float]] = field(
        default_factory=lambda: {task: [] for task in TASKS}
    )


def build_accelerator(
    output_dir: str, gradient_accumulation_steps: int = 1
) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Long timeout so rank-0-only setup doesn't trip the 10-min NCCL barrier.
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )


def checkpoint_epoch_offset(init_from_checkpoint: Optional[str]) -> int:
    """Epoch number baked into an `epoch-N.pt` filename, else 0.

    Continuing a run writes `epoch-{offset + n}.pt`, so resuming into a shared
    output_dir doesn't overwrite the checkpoint it started from.
    """
    if not init_from_checkpoint:
        return 0
    match = re.match(r"epoch-(\d+)", os.path.basename(init_from_checkpoint))
    return int(match.group(1)) if match else 0


def setup_model(cfg: RunConfig) -> Tuple[MultiTaskRoberta, Dict[str, Any]]:
    """Build the model, restoring from `cfg.init_from_checkpoint` if set, and return it
    with the head sizes it was built with.
    """
    # Head sizes come from the checkpoint, or `load_state_dict` fails on a shape.
    head_sizes: Dict[str, Any] = {
        "num_themes": cfg.theme_count,
        "num_tones": 1,
    }

    checkpoint = None
    if cfg.init_from_checkpoint:
        checkpoint = torch.load(
            cfg.init_from_checkpoint, map_location="cpu", weights_only=False
        )
        state = checkpoint["model_state_dict"]
        saved = checkpoint.get("config", {}) or {}
        head_sizes = {
            "num_themes": saved.get("num_themes", state["theme_head.weight"].shape[0]),
            "num_tones": saved.get("num_tones", state["tone_head.weight"].shape[0]),
        }

    model = MultiTaskRoberta(
        **head_sizes, loss_weights=cfg.loss_weights.as_dict()
    )

    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        log(
            f"restored {cfg.init_from_checkpoint} "
            f"({len(missing)} missing / {len(unexpected)} unexpected keys)"
        )

    # Gradient checkpointing is OFF; uncomment for ~5-10x less activation memory at
    # ~20-30% more step time. Stays non-reentrant -- see gradient_checkpointing_enable.
    # model.backbone.config.use_cache = False
    # model.gradient_checkpointing_enable()

    return model, head_sizes


def prepare_run(cfg: RunConfig, accelerator: Accelerator) -> Tuple[PreparedRun, int, Dict[str, Any]]:
    """Build model, data, optimizer and schedule, and hand them to Accelerate.

    Also returns the un-sharded steps-per-epoch (measured before `prepare`
    shards the loader) and the head sizes, both only used for reporting.
    """
    args = cfg.train_args

    model, head_sizes = setup_model(cfg)
    model.to(accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataloader, active_tasks = build_dataloader(
        tok=tokenizer,
        task_spec=cfg.task_spec,
        args=args,
        tasks_to_build=cfg.tasks,
    )

    steps_per_epoch = len(dataloader)
    max_steps_per_epoch = steps_per_epoch // (
        accelerator.num_processes * args.gradient_accumulation_steps
    )
    if args.log_every is None:
        args.log_every = max(1, max_steps_per_epoch // 15)
    total_steps = args.num_epochs * max_steps_per_epoch
    if args.max_steps is not None:
        if args.max_steps > total_steps:
            log(
                f"   max_steps={args.max_steps:,} exceeds the {total_steps:,} "
                f"optimizer steps {args.num_epochs} epoch(s) of this corpus "
                "provide; training will stop when the data runs out."
            )
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr, weight_decay=0.01, fused=True
    )
    schedule_scale = accelerator.num_processes
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio) * schedule_scale,
        num_training_steps=total_steps * schedule_scale,
    )

    dataloader = accelerator.prepare(dataloader)
    model, scheduler, optimizer = accelerator.prepare(model, scheduler, optimizer)

    run = PreparedRun(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        active_tasks=active_tasks,
        max_steps_per_epoch=max_steps_per_epoch,
        total_steps=total_steps,
    )
    return run, steps_per_epoch, head_sizes


def run_summary_lines(
    cfg: RunConfig,
    config_path: str,
    accelerator: Accelerator,
    head_sizes: Dict[str, Any],
    steps_per_epoch: int,
    run: PreparedRun,
) -> List[str]:
    """The run's settings, one 'key : value' line each.

    Printed to stdout and written verbatim as the training_log.txt header, so
    the two never drift apart.
    """
    args = cfg.train_args
    spec = cfg.task_spec
    lines = [
        "MULTI-TASK PRETRAINING",
        f"started         : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"config          : {config_path}",
        f"output dir      : {cfg.output_dir}",
        f"model           : {args.model_name}",
        f"checkpoint      : {cfg.init_from_checkpoint or 'none (fresh roberta-base)'}",
        f"dataset         : {spec.dataset_name}",
        f"tasks           : {', '.join(cfg.tasks)}",
        f"heads           : themes={head_sizes['num_themes']}, "
        f"tones={head_sizes['num_tones']}",
        f"processes       : {accelerator.num_processes} "
        f"({accelerator.distributed_type}) on {accelerator.device}, "
        f"mixed precision {accelerator.mixed_precision}",
        f"batch size      : {args.batch_size}/process x "
        f"{accelerator.num_processes} process(es) x "
        f"{args.gradient_accumulation_steps} accumulated = "
        f"{args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
        f" examples/optimizer step, {spec.max_triplet_samples} triplets/batch",
        f"epochs          : {args.num_epochs} x {run.max_steps_per_epoch:,} "
        f"= {run.total_steps:,} optimizer steps"
        f"{f'  [capped by max_steps={args.max_steps:,}]' if args.max_steps is not None else ''}",
        f"learning rate   : {cfg.base_lr} (warmup {args.warmup_ratio:.0%} = "
        f"{int(run.total_steps * args.warmup_ratio):,} steps, linear decay)",
        f"loss weights    : {cfg.loss_weights.as_dict()}"
        f"{'' if cfg.loss_weights.is_default() else '  [OVERRIDDEN]'}",
        f"subsampled rows : {spec.require_nonempty_themes_and_tone}",
        f"batching        : one shared loader, {len(run.dataloader.dataset):,} samples, "
        f"{steps_per_epoch:,} steps/epoch (all objectives see the same rows)",
    ]
    for name in cfg.tasks:
        if name in run.active_tasks:
            lines.append(f"  {name:8s}: collated")
        else:
            lines.append(f"  {name:8s}: no collator (skipped)")
    return lines


def build_diagnostics(
    cfg: RunConfig,
    accelerator: Accelerator,
    run: PreparedRun,
    config_path: str,
) -> Optional[GradientDiagnostics]:
    """Construct the gradient diagnostic, or None when it is disabled.

    Must run after `accelerator.prepare` so it can unwrap the model and hold
    references to the shared-representation parameters.
    """
    if not cfg.gradient_diagnostics.enabled:
        return None
    return GradientDiagnostics(
        accelerator=accelerator,
        model=run.model,
        config=cfg.gradient_diagnostics,
        output_dir=cfg.output_dir,
        train_batch_size=cfg.train_args.batch_size,
        # Collator tasks, so a task that produced nothing still records a zero gradient.
        active_tasks=run.active_tasks,
        checkpoint=cfg.init_from_checkpoint or "none",
        seed=SEED,
        gradient_accumulation_steps=cfg.train_args.gradient_accumulation_steps,
        total_steps=run.total_steps,
        # Recorded verbatim in the JSON/TXT so a reader can reproduce the run.
        metadata={
            "config_path": config_path,
            "model_name": cfg.train_args.model_name,
            "dataset": cfg.task_spec.dataset_name,
            "tasks": ",".join(cfg.tasks),
            "max_triplet_samples": cfg.task_spec.max_triplet_samples,
            "require_nonempty_themes_and_tone": cfg.task_spec.require_nonempty_themes_and_tone,
            "base_lr": cfg.base_lr,
        },
    )


def next_batch(iterator: Any, dataloader: Any) -> Tuple[Dict[str, Any], Any]:
    """Next shared batch, restarting the loader if it ran out mid-epoch.

    Returns the batch and the (possibly restarted) iterator, so the caller keeps
    holding the live one.
    """
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def build_combined_batch(
    tasks: Sequence[str], batches: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge one shared batch's per-task sub-batches into a model kwargs dict, renaming
    collator keys to the prefixed ones `forward` dispatches on.
    """
    # Every task here sees the same articles; tasks that signalled `_skip` are absent.
    combined: Dict[str, Any] = {}
    for task_name in tasks:
        batch = batches.get(task_name)
        if batch is None:
            continue
        for source_key, model_key in TASK_BATCH_KEYS[task_name].items():
            combined[model_key] = batch[source_key]
    return combined


def train_one_epoch(
    epoch: int,
    cfg: RunConfig,
    run: PreparedRun,
    state: TrainingState,
    logger: TrainingLogger,
    diagnostics: Optional[GradientDiagnostics],
) -> None:
    """Run one epoch, mutating `state`; `epoch` is 1-based and for display only."""
    # The iterator is rebuilt here, so each epoch reshuffles once for all objectives.
    args = cfg.train_args
    accelerator = run.accelerator
    diagnostic_only = (
        cfg.gradient_diagnostics.enabled and cfg.gradient_diagnostics.diagnostic_only
    )
    diagnostics_reported = False

    batch_iterator = iter(run.dataloader)

    epoch_start_time = time.time()
    steps_this_epoch = 0
    group_start = time.time()

    micro_budget = state.micro_step + run.max_steps_per_epoch * (
        args.gradient_accumulation_steps + 1
    )

    while (
        steps_this_epoch < run.max_steps_per_epoch
        and state.step < run.total_steps
        and state.micro_step < micro_budget
    ):
        stepped = False

        batches, batch_iterator = next_batch(batch_iterator, run.dataloader)
        combined_batch = build_combined_batch(cfg.tasks, batches)
        if not combined_batch:
            state.micro_step += 1
            continue

        with accelerator.accumulate(run.model):
            outputs = run.model(**combined_batch)
            total_loss = outputs.get("loss")

            # Runs after forward, before backward; autograd.grad leaves .grad and the graph alone.
            if diagnostics is not None and diagnostics.should_measure(state.step):
                # The step is stamped on every measurement, giving a periodic run its x-axis.
                diagnostics.set_train_step(state.step)
                diagnostics.record_microbatch(outputs, combined_batch)

                if diagnostic_only:
                    # No backward or step: every virtual batch measures the same model state.
                    del outputs, total_loss, combined_batch
                    state.step += 1
                    state.micro_step += 1
                    steps_this_epoch += 1
                    if diagnostics.is_complete():
                        return
                    continue

                if diagnostics.is_complete() and not diagnostics_reported:
                    # Diagnostics ran alongside training; report once and carry on.
                    diagnostics.report()
                    diagnostics_reported = True

            if total_loss is not None:
                accelerator.backward(total_loss)

                if accelerator.is_main_process:
                    for task, loss_key in TASK_LOSS_KEYS.items():
                        value = outputs.get(loss_key)
                        if value is not None:
                            state.task_losses[task].append(value.item())
                        weighted = outputs.get(TASK_WEIGHTED_LOSS_KEYS[task])
                        if weighted is not None:
                            state.weighted_losses[task].append(weighted.item())

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(run.model.parameters(), max_norm=1.0)

            stepped = accelerator.sync_gradients
            run.optimizer.step()
            run.scheduler.step()
            run.optimizer.zero_grad()

        state.micro_step += 1
        if not stepped:
            # Mid-accumulation: no optimizer step happened
            continue

        state.step += 1
        steps_this_epoch += 1
        state.step_times.append(time.time() - group_start)
        group_start = time.time()

        if accelerator.is_main_process and state.step % args.log_every == 0:
            interval_losses = {
                task: sum(values) / len(values)
                for task, values in state.task_losses.items()
                if values
            }
            interval_weighted = {
                task: sum(values) / len(values)
                for task, values in state.weighted_losses.items()
                if values
            }
            if interval_losses:
                for task in interval_losses:
                    state.task_losses[task] = []
                for task in interval_weighted:
                    state.weighted_losses[task] = []
                logger.log_interval(
                    step=state.step,
                    epoch=epoch,
                    num_epochs=args.num_epochs,
                    total_steps=run.total_steps,
                    lr=run.scheduler.get_last_lr()[0],
                    progress=training_progress(
                        state.started_at,
                        epoch_start_time,
                        state.step_times,
                        state.step,
                        steps_this_epoch,
                        run.max_steps_per_epoch,
                        run.total_steps,
                    ),
                    task_losses=interval_losses,
                    weighted_losses=interval_weighted,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to run YAML config")
    cli = parser.parse_args()
    cfg = load_run_config(cli.config)

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(SEED)

    accelerator = build_accelerator(
        cfg.output_dir, cfg.train_args.gradient_accumulation_steps
    )

    # Fail fast: the diagnostic's autograd.grad must not run on a DDP-wrapped model.
    diag_cfg = cfg.gradient_diagnostics
    if diag_cfg.enabled and accelerator.num_processes != 1:
        raise RuntimeError(
            "Gradient diagnostics must be run with a single Accelerate process."
        )
    diagnostic_only = diag_cfg.enabled and diag_cfg.diagnostic_only

    run, steps_per_epoch, head_sizes = prepare_run(cfg, accelerator)

    summary = run_summary_lines(
        cfg, cli.config, accelerator, head_sizes, steps_per_epoch, run
    )
    log("=" * 78)
    for line in summary:
        log(line)
    log("=" * 78)

    # A diagnostic-only run leaves earlier logs alone and writes its own files.
    logger = TrainingLogger(
        cfg.output_dir,
        TASKS,
        summary,
        enabled=accelerator.is_main_process and not diagnostic_only,
    )
    diagnostics = build_diagnostics(cfg, accelerator, run, cli.config)

    epoch_offset = checkpoint_epoch_offset(cfg.init_from_checkpoint)
    state = TrainingState()
    run.model.train()
    log(f"\nSTARTING TRAINING: {cfg.train_args.num_epochs} epoch(s), {run.total_steps:,} steps\n")

    for epoch in range(1, cfg.train_args.num_epochs + 1):
        epoch_start_step = state.step
        train_one_epoch(epoch, cfg, run, state, logger, diagnostics)

        if diagnostic_only:
            break

        if accelerator.is_main_process:
            checkpoint_path = f"{cfg.output_dir}/epoch-{epoch_offset + epoch}.pt"
            accelerator.unwrap_model(run.model).save_checkpoint(checkpoint_path)
            logger.note(
                f"completed epoch {epoch}: {state.step - epoch_start_step:,} steps "
                f"({state.step:,} total) -> {checkpoint_path}"
            )

    if diagnostic_only and diagnostics is not None:
        diagnostics.report()
        log(
            f"\nDiagnostic-only run: {state.step} forward passes, no optimizer steps "
            "taken. Training objective and weights are unchanged."
        )
        accelerator.end_training()
        return

    if diagnostics is not None and diagnostics.periodic:
        diagnostics.set_train_step(state.step)
        diagnostics.report()

    accelerator.end_training()

    elapsed = str(timedelta(seconds=int(time.time() - state.started_at)))
    log("")
    log("=" * 78)
    log(
        f"TRAINING COMPLETE: {cfg.train_args.num_epochs} epoch(s), "
        f"{state.step:,} optimizer steps ({state.micro_step:,} forwards), "
        f"{elapsed} elapsed"
    )
    log(f"checkpoints, training_log.txt and losses.tsv in {cfg.output_dir}")
    log("=" * 78)


if __name__ == "__main__":
    main()
