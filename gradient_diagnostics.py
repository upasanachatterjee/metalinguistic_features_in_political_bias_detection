"""Temporary diagnostic: per-task gradient magnitude on the shared representation, the
pairwise conflict between those gradients, and the triplet geometry behind one of them.

Norms are `||(1/K) sum_k g_k||`, never `(1/K) sum_k ||g_k||`
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Numerical floor for cosine denominators.
EPS = 1e-12

# Logical task name -> exact key returned by MultiTaskRoberta.forward().
TASK_LOSS_KEYS: Dict[str, str] = {
    "triplet": "triplet_loss",
    "themes": "theme_loss",
    "tone": "tone_loss",
    "mlm": "mlm_loss",
}

# Task -> batch key counted per microbatch, to confirm K x batch_size examples.
TASK_BATCH_KEYS: Dict[str, str] = {
    "triplet": "triplet_a_ids",  # anchors; each anchor also pulls a pos + neg
    "themes": "theme_input_ids",
    "tone": "tone_input_ids",
    "mlm": "mlm_input_ids",
}

# Reference task for suggested weights: w_mlm is pinned to 1.0, the rest relative.
REFERENCE_TASK = "mlm"

# Clamp: a suggested 50x reweighting means the task is wrong, not the weight.
MIN_SUGGESTED_WEIGHT = 0.2
MAX_SUGGESTED_WEIGHT = 5.0

# Rounded DOWN onto this grid, so "conservative" never overstates the change.
WEIGHT_ROUNDING_GRID = 0.05

# Repeated cosines below this are worth investigating; one negative is not.
CONFLICT_COSINE = -0.3

# Triplet geometry keys carried through from model._triplet_stats.
TRIPLET_STAT_KEYS = (
    "num_triplets",
    "active_fraction",
    "mean_positive_distance",
    "mean_negative_distance",
    "mean_violation",
)

# Every row carries the knob settings, so TSVs from separate runs concatenate.
TSV_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "run_label",
    "parameter_scope",
    "num_diag_params",
    "train_batch_size",
    "microbatches_per_virtual_batch",
    "examples_per_virtual_batch",
    "checkpoint",
    "train_step",
    "loss_weights",
    "virtual_batch",
    "metric",
    "task_a",
    "task_b",
    "value",
)

TSV_HEADER_COMMENT = """\
# Multi-task gradient diagnostics, tidy long format (one measurement per row).
# metric values:
#   grad_norm   ||mean gradient|| for task_a over one virtual batch
#   cosine      cosine(mean grad task_a, mean grad task_b) over one virtual batch
#   loss        mean RAW (unweighted) loss for task_a over one virtual batch
#   examples    examples fed to task_a during one virtual batch
#   microbatches_with_task   microbatches (of K) that produced a loss for task_a
#   triplet_*   triplet geometry, averaged over the virtual batch's microbatches
#   diag_secs   seconds spent in torch.autograd.grad for the virtual batch
#   wall_secs   wall-clock seconds for the virtual batch (fwd + diagnostic)
#   peak_mem_mib  peak CUDA memory during the virtual batch, MiB (CUDA only)
# Gradient norms are ||(1/K) sum_k g_k||, NOT (1/K) sum_k ||g_k||.
# Microbatches with no loss for a task contribute a zero vector, and the sum is
# still divided by K (this mirrors an idle DDP rank).
# Load with: pd.read_csv(path, sep="\\t", comment="#")
"""


@dataclass
class GradientDiagnosticsConfig:
    """YAML knobs for the gradient diagnostic. Set
    `microbatches_per_virtual_batch = target_global_batch_size // batch_size`; the
    constructor raises on a mismatch, since every number is relative to that geometry.
    """

    enabled: bool = False
    microbatches_per_virtual_batch: int = 8
    num_virtual_batches: int = 12
    parameter_scope: str = "final_encoder_layer"
    diagnostic_only: bool = True
    diagnostics_every_n_virtual_batches: Optional[int] = None
    # Global batch size the diagnostic emulates: 8 GPUs x 32 per GPU.
    target_global_batch_size: int = 256
    # Caps retained examples: theme logits are 2000 floats each.
    max_metric_examples: int = 4096
    run_label: Optional[str] = None
    records_path: Optional[str] = None


def get_diagnostic_parameters(
    base_model: torch.nn.Module, scope: str
) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """The shared-representation parameters to differentiate against, from an
    unwrapped `base_model`. `full_backbone` costs ~440MB of CPU float32 per task.
    """
    # Typed as Any: nn.Module.__getattr__ returns Tensor | Module.
    backbone: Any = base_model.backbone
    if scope == "final_encoder_layer":
        modules: Sequence[torch.nn.Module] = [backbone.encoder.layer[-1]]
    elif scope == "full_backbone":
        modules = [backbone]
    elif scope.startswith("last_") and scope.endswith("_encoder_layers"):
        n = int(scope.split("_")[1])
        modules = list(backbone.encoder.layer[-n:])
    else:
        raise ValueError(
            f"Unknown parameter_scope '{scope}'. Expected 'final_encoder_layer', "
            "'full_backbone' or 'last_N_encoder_layers'."
        )

    params: List[torch.nn.Parameter] = []
    names: List[str] = []
    for module in modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                params.append(param)
                names.append(name)
    return params, names


def scope_tag(scope: str) -> str:
    """Short form of a parameter scope, for run labels."""
    if scope == "final_encoder_layer":
        return "final"
    if scope == "full_backbone":
        return "full"
    if scope.startswith("last_") and scope.endswith("_encoder_layers"):
        return f"last{scope.split('_')[1]}"
    return scope


def checkpoint_tag(checkpoint: Optional[str]) -> str:
    """Short form of the starting checkpoint: `init` for a fresh roberta-base,
    otherwise the filename stem, so `epoch-1.pt` becomes `epoch-1`.
    """
    if not checkpoint or checkpoint == "none":
        return "init"
    return os.path.splitext(os.path.basename(checkpoint))[0]


def count_batch_examples(combined_batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Per-task example counts for one microbatch (for verification output)."""
    counts: Dict[str, int] = {}
    for task, key in TASK_BATCH_KEYS.items():
        tensor = combined_batch.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) > 0:
            counts[task] = int(tensor.shape[0])
    return counts


def conservative_weight(weight: float) -> float:
    """Clamp a suggested weight and round it DOWN onto WEIGHT_ROUNDING_GRID.

    Rounding down rather than to-nearest means the printed suggestion never
    proposes a bigger correction than the measurement supports.
    """
    if not math.isfinite(weight):
        return 1.0
    clamped = min(max(weight, MIN_SUGGESTED_WEIGHT), MAX_SUGGESTED_WEIGHT)
    grid = WEIGHT_ROUNDING_GRID
    rounded = math.floor(clamped / grid + 1e-9) * grid
    return round(max(rounded, MIN_SUGGESTED_WEIGHT), 4)


def git_commit() -> Optional[str]:
    """Current commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def describe(values: Sequence[float]) -> Dict[str, float]:
    """Median-first summary of one measurement series.

    Median leads because a dozen virtual batches is few enough that one outlier
    (a batch where triplet mining found two hard triplets, say) moves the mean
    a long way.
    """
    ordered = sorted(values)
    n = len(ordered)
    mean = statistics.fmean(ordered)
    std = statistics.pstdev(ordered) if n > 1 else 0.0

    def percentile(q: float) -> float:
        if n == 1:
            return ordered[0]
        position = q * (n - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": n,
        "median": statistics.median(ordered),
        "mean": mean,
        "std": std,
        "p5": percentile(0.05),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        # Dimensionless spread: which gradient is *unstable*, not merely large.
        "cv": (std / abs(mean)) if abs(mean) > EPS else float("nan"),
    }


@dataclass
class _VirtualBatchAccumulator:
    """Per-virtual-batch state, reset after every finalize."""

    grad_sums: Dict[str, torch.Tensor] = field(default_factory=dict)
    task_microbatches: Dict[str, int] = field(default_factory=dict)
    loss_sums: Dict[str, float] = field(default_factory=dict)
    example_counts: Dict[str, int] = field(default_factory=dict)
    triplet_stats: List[Dict[str, float]] = field(default_factory=list)
    triplet_empty_microbatches: int = 0
    microbatches: int = 0
    diag_seconds: float = 0.0
    started: float = field(default_factory=time.perf_counter)


class GradientDiagnostics:
    """Accumulates per-task gradient vectors over virtual batches and reports."""

    def __init__(
        self,
        accelerator,
        model: torch.nn.Module,
        config: GradientDiagnosticsConfig,
        output_dir: Optional[str] = None,
        train_batch_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        active_tasks: Optional[Sequence[str]] = None,
        checkpoint: Optional[str] = None,
        seed: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        total_steps: Optional[int] = None,
    ) -> None:
        if accelerator.num_processes != 1:
            raise RuntimeError(
                "Gradient diagnostics must be run with a single Accelerate process, "
                f"got {accelerator.num_processes}. torch.autograd.grad on a "
                "DDP-wrapped module does not produce the per-task gradients this "
                "measures. Launch with:\n"
                "  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 "
                "pretraining.py --config CONFIG.yaml"
            )
        if train_batch_size is None:
            raise ValueError(
                "train_batch_size is required: the diagnostic verifies "
                "batch_size x microbatches_per_virtual_batch against "
                "target_global_batch_size."
            )

        k = config.microbatches_per_virtual_batch
        if k < 1:
            raise ValueError(f"microbatches_per_virtual_batch must be >= 1, got {k}.")
        effective = train_batch_size * k
        if effective != config.target_global_batch_size:
            raise ValueError(
                "Diagnostic batch geometry does not match normal training.\n"
                f"  train_args.batch_size                        = {train_batch_size}\n"
                f"  gradient_diagnostics.microbatches_per_virtual_batch = {k}\n"
                f"  effective diagnostic batch                   = {effective}\n"
                f"  target_global_batch_size                     = "
                f"{config.target_global_batch_size}\n"
                "Set microbatches_per_virtual_batch = "
                f"{config.target_global_batch_size} // {train_batch_size} = "
                f"{config.target_global_batch_size // train_batch_size} "
                "(or fix target_global_batch_size to the real "
                "num_gpus x batch_size)."
            )

        interval = config.diagnostics_every_n_virtual_batches
        if interval is not None:
            if interval < 1:
                raise ValueError(
                    "diagnostics_every_n_virtual_batches must be >= 1, got "
                    f"{interval}."
                )
            if config.diagnostic_only:
                raise ValueError(
                    "diagnostics_every_n_virtual_batches needs diagnostic_only: "
                    "false. Periodic snapshots exist to show how the objectives "
                    "change AS THE MODEL TRAINS; diagnostic_only takes no "
                    "optimizer steps, so every snapshot would measure the same "
                    "frozen weights. For single-point measurements of a trained "
                    "model, keep diagnostic_only: true and set "
                    "init_from_checkpoint instead."
                )

        self.accelerator = accelerator
        self.config = config
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.metadata = dict(metadata or {})
        self.checkpoint = checkpoint or "none"
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_steps = total_steps
        self.examples_per_virtual_batch = effective
        self.virtual_batch_is_optimizer_step = gradient_accumulation_steps == k

        base_model = accelerator.unwrap_model(model)
        self.parameters, param_names = get_diagnostic_parameters(
            base_model, config.parameter_scope
        )
        self.num_params = sum(p.numel() for p in self.parameters)
        self.applied_loss_weights = dict(getattr(base_model, "loss_weights", {}) or {})

        # A task with no loss records norm 0, or it drops out of its own median.
        self.tracked_tasks: List[str] = [
            task for task in TASK_LOSS_KEYS if active_tasks and task in active_tasks
        ]
        # Ask the model for triplet geometry only while the diagnostic runs.
        base_model.collect_triplet_stats = True

        self.started_at = datetime.now()
        self.run_id = self.started_at.strftime("%Y%m%dT%H%M%S")
        self.run_label = config.run_label or (
            f"scope-{scope_tag(config.parameter_scope)}"
            f"_bs{train_batch_size}_k{k}"
            f"_global{effective}_{checkpoint_tag(self.checkpoint)}"
        )

        self._acc = _VirtualBatchAccumulator()

        # Per-snapshot state; `_reset_window` keeps snapshots independent.
        self.norm_history: Dict[str, List[float]] = {}
        self.cosine_history: Dict[Tuple[str, str], List[float]] = {}
        self.loss_history: Dict[str, List[float]] = {}
        self.triplet_history: Dict[str, List[float]] = {}
        self.virtual_batches: List[Dict[str, Any]] = []
        self._tone_preds: List[float] = []
        self._tone_targets: List[float] = []
        self._theme_logits: List[Any] = []
        self._theme_labels: List[Any] = []
        self._task_metrics: Optional[Dict[str, Any]] = None
        self.completed_virtual_batches = 0
        self.window_virtual_batches = 0
        self.snapshots: List[Dict[str, Any]] = []
        self.current_train_step = 0
        self.window_start_step = 0

        self._warned: set = set()
        self._lines: List[str] = []
        self._rows: List[Tuple[Any, ...]] = []
        self._rows_flushed = 0

        self._emit_header(param_names)

    def _emit_header(self, param_names: Sequence[str]) -> None:
        config = self.config
        k = config.microbatches_per_virtual_batch

        self._emit("=" * 78)
        self._emit("GRADIENT DIAGNOSTICS (temporary; training objective unchanged)")
        self._emit("=" * 78)

        # Printed first: every other number in the report is conditional on it.
        self._emit("  diagnostic batch size:         "
                   f"{self.train_batch_size:>10}")
        self._emit(f"  microbatches per virtual batch:{k:>10}")
        self._emit("  effective diagnostic batch:    "
                   f"{self.examples_per_virtual_batch:>10}")
        self._emit("  normal training global batch:  "
                   f"{config.target_global_batch_size:>10}")
        self._emit(
            "  -> K sequential local batches, gradients averaged; NOT one "
            "physical batch of "
            f"{self.examples_per_virtual_batch}."
        )
        if self.config.diagnostics_every_n_virtual_batches is not None:
            interval = self.config.diagnostics_every_n_virtual_batches
            self._emit(f"  gradient accumulation steps:   {self.gradient_accumulation_steps:>10}")
            if self.virtual_batch_is_optimizer_step:
                self._emit(
                    f"  snapshot interval:             {interval:>10} virtual "
                    f"batches = {interval} optimizer steps"
                )
                self._emit(
                    "  -> one virtual batch IS one optimizer step: the measured "
                    "gradient is the one the optimizer used."
                )
            else:
                self._emit(
                    f"  snapshot interval:             {interval:>10} virtual "
                    f"batches = {interval * k} microbatches"
                )
                self._emit(
                    f"  WARNING: gradient_accumulation_steps="
                    f"{self.gradient_accumulation_steps} != K={k}, so a virtual "
                    "batch is NOT an optimizer step and the snapshot interval "
                    "does not read as training steps. Set "
                    f"train_args.gradient_accumulation_steps: {k} to align them."
                )
            self._emit(
                f"  measurements per snapshot:     "
                f"{self.config.num_virtual_batches:>10}"
            )
        self._emit("-" * 78)

        self._emit(f"  run id                       : {self.run_id}")
        self._emit(f"  run label                    : {self.run_label}")
        self._emit(f"  checkpoint                   : {self.checkpoint}")
        self._emit(f"  seed                         : {self.seed}")
        self._emit(f"  git commit                   : {git_commit() or 'unavailable'}")
        self._emit(f"  parameter scope              : {config.parameter_scope}")
        self._emit(
            f"  diagnosed parameters         : {len(self.parameters)} tensors, "
            f"{self.num_params:,} scalars"
        )
        self._emit(
            f"    first/last tensor          : {param_names[0]} ... {param_names[-1]}"
        )
        self._emit(f"  virtual batches to collect   : {config.num_virtual_batches}")
        self._emit(f"  tasks tracked                : {', '.join(self.tracked_tasks)}")
        self._emit(f"  diagnostic_only              : {config.diagnostic_only}")
        self._emit(
            f"  loss weights in effect       : {self.applied_loss_weights or 'n/a'}"
        )
        for key, value in self.metadata.items():
            self._emit(f"  {key:<29}: {value}")

        if getattr(self.accelerator, "mixed_precision", "no") == "fp16":
            self._emit(
                "  WARNING: mixed_precision=fp16. Diagnostic gradients are computed "
                "unscaled (no GradScaler), so small values may underflow. Prefer "
                "launching the diagnostic with --mixed_precision no or bf16."
            )
        if config.diagnostic_only:
            self._emit(
                "  Optimizer/scheduler steps are SKIPPED: all virtual batches "
                "measure the same model state."
            )
            self._emit(
                "  Model stays in train() mode, so dropout noise is included "
                "exactly as it is during training."
            )
        else:
            self._emit(
                "  NOTE: normal training steps still run, so model parameters "
                "change between virtual batches."
            )
        self._emit("-" * 78)

    @property
    def periodic(self) -> bool:
        """True when snapshots are taken during training rather than once."""
        return self.config.diagnostics_every_n_virtual_batches is not None

    def is_complete(self) -> bool:
        """Whether the one-shot measurement has collected everything it needs.

        Meaningless in periodic mode, where measurement continues for as long as
        training does -- ``should_measure`` decides when to record instead.
        """
        if self.periodic:
            return False
        return self.completed_virtual_batches >= self.config.num_virtual_batches

    def should_measure(self, optimizer_step: int) -> bool:
        """Whether this optimizer step falls inside a measurement window.
        """
        if not self.periodic:
            return not self.is_complete()
        interval = self.config.diagnostics_every_n_virtual_batches
        assert interval is not None  # guarded by self.periodic
        window = self.config.num_virtual_batches
        if (optimizer_step % interval) < window:
            return True
        return self.total_steps is not None and optimizer_step >= (
            self.total_steps - window
        )

    def set_train_step(self, optimizer_step: int) -> None:
        """Tell the diagnostic which optimizer step it is currently measuring.

        The first step of a window names the snapshot, so a snapshot reported at
        step 400 is one that began measuring at step 400 rather than one that
        happened to finish there.
        """
        if self.window_virtual_batches == 0 and self._acc.microbatches == 0:
            self.window_start_step = optimizer_step
        self.current_train_step = optimizer_step

    def record_microbatch(
        self,
        outputs: Dict[str, torch.Tensor],
        combined_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Compute per-task gradients for one microbatch, after the forward pass and
        before `accelerator.backward`.
        """
        # retain_graph=True so the combined backward still runs on the same graph.
        if self.is_complete():
            return

        started = time.perf_counter()
        acc = self._acc

        if combined_batch is not None:
            for task, count in count_batch_examples(combined_batch).items():
                acc.example_counts[task] = acc.example_counts.get(task, 0) + count

        self._record_triplet_stats(outputs, combined_batch)
        self._record_predictions(outputs, combined_batch)

        for task, loss_key in TASK_LOSS_KEYS.items():
            loss = outputs.get(loss_key)
            if loss is None:
                # An idle DDP rank: _finalize still divides by K, so this is a zero vector.
                continue
            if not torch.is_tensor(loss) or not loss.requires_grad:
                self._warn_once(
                    task,
                    f"  [skip] {task}: '{loss_key}' has no grad_fn; not differentiable.",
                )
                continue

            grads = torch.autograd.grad(
                loss,
                self.parameters,
                retain_graph=True,
                allow_unused=True,
            )
            vector = self._flatten(task, grads)
            if vector is None:
                continue

            if task in acc.grad_sums:
                acc.grad_sums[task] += vector
            else:
                acc.grad_sums[task] = vector
            acc.task_microbatches[task] = acc.task_microbatches.get(task, 0) + 1
            acc.loss_sums[task] = acc.loss_sums.get(task, 0.0) + float(loss.item())
            # Drop the autograd tensors now so no graph outlives this iteration.
            del grads, vector

        acc.diag_seconds += time.perf_counter() - started
        acc.microbatches += 1
        if acc.microbatches >= self.config.microbatches_per_virtual_batch:
            self._finalize_virtual_batch()

    def _record_triplet_stats(
        self,
        outputs: Dict[str, Any],
        combined_batch: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Per-microbatch triplet geometry, including empty microbatches.

        Recorded per *local* batch, because that is the unit triplets are mined
        in: mining happens inside the collator, over one batch of 32 rows, on
        each rank independently.
        """
        stats = outputs.get("triplet_stats")
        if stats:
            self._acc.triplet_stats.append(dict(stats))
            return

        # Collator signalled `_skip`: no valid triplets at all, so record zero.
        if combined_batch is not None and "triplet_a_ids" not in combined_batch:
            self._acc.triplet_empty_microbatches += 1
            self._acc.triplet_stats.append({"num_triplets": 0})

    def _record_predictions(
        self,
        outputs: Dict[str, Any],
        combined_batch: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Stash tone/theme predictions for the quality read-outs."""
        if combined_batch is None:
            return
        cap = self.config.max_metric_examples

        tone_logits = outputs.get("tone_logits")
        tone_labels = combined_batch.get("tone_labels")
        if tone_logits is not None and tone_labels is not None:
            if len(self._tone_preds) < cap:
                self._tone_preds.extend(
                    tone_logits.detach().reshape(-1).float().cpu().tolist()
                )
                self._tone_targets.extend(
                    tone_labels.detach().reshape(-1).float().cpu().tolist()
                )

        theme_logits = outputs.get("theme_logits")
        theme_labels = combined_batch.get("theme_labels")
        if theme_logits is not None and theme_labels is not None:
            kept = sum(chunk.shape[0] for chunk in self._theme_logits)
            if kept < cap:
                self._theme_logits.append(
                    theme_logits.detach().float().cpu().numpy()
                )
                self._theme_labels.append(
                    theme_labels.detach().float().cpu().numpy()
                )

    def _flatten(
        self, task: str, grads: Iterable[Optional[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Detach, flatten and move one task's gradients to a CPU float32 vector.

        Parameter order follows ``self.parameters``, which is fixed at
        construction, so the same index means the same scalar for every task and
        every microbatch -- that is what makes the cosines meaningful.
        """
        chunks: List[torch.Tensor] = []
        any_grad = False
        for grad, param in zip(grads, self.parameters):
            if grad is None:
                chunks.append(torch.zeros(param.numel(), dtype=torch.float32))
            else:
                any_grad = True
                chunks.append(grad.detach().reshape(-1).to(torch.float32).cpu())
        if not any_grad:
            self._warn_once(
                f"{task}-unused",
                f"  [skip] {task}: loss does not depend on the diagnosed parameters.",
            )
            return None
        return torch.cat(chunks)

    def _finalize_virtual_batch(self) -> None:
        """Average the accumulated gradients, record statistics, reset state."""
        k = self.config.microbatches_per_virtual_batch
        acc = self._acc
        self.completed_virtual_batches += 1
        self.window_virtual_batches += 1
        index = self.window_virtual_batches
        wall_seconds = time.perf_counter() - acc.started

        # ||mean gradient||, not mean of norms; always over K, not over the tasked ones.
        mean_grads: Dict[str, torch.Tensor] = {
            task: grad_sum / float(k) for task, grad_sum in acc.grad_sums.items()
        }
        for task in mean_grads:
            if task not in self.tracked_tasks:
                self.tracked_tasks.append(task)

        # Fixed order; an absent task records 0.0 rather than skewing its own median.
        norms: Dict[str, float] = {}
        for task in self.tracked_tasks:
            mean_grad = mean_grads.get(task)
            norms[task] = (
                float(torch.linalg.vector_norm(mean_grad).item())
                if mean_grad is not None
                else 0.0
            )

        record: Dict[str, Any] = {
            "virtual_batch": index,
            "grad_norm": {},
            "cosine": {},
            "loss": {},
            "microbatches_with_task": dict(acc.task_microbatches),
            "examples": dict(acc.example_counts),
            "microbatches": acc.microbatches,
            "diag_seconds": round(acc.diag_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }

        for task, norm in norms.items():
            self.norm_history.setdefault(task, []).append(norm)
            record["grad_norm"][task] = norm
            self._row(index, "grad_norm", task, "", norm)

            contributing = acc.task_microbatches.get(task, 0)
            self._row(index, "microbatches_with_task", task, "", contributing)
            if contributing:
                mean_loss = acc.loss_sums[task] / contributing
                self.loss_history.setdefault(task, []).append(mean_loss)
                record["loss"][task] = mean_loss
                self._row(index, "loss", task, "", mean_loss)

        for task_a, task_b in combinations(sorted(mean_grads), 2):
            g_a, g_b = mean_grads[task_a], mean_grads[task_b]
            norm_a = torch.linalg.vector_norm(g_a)
            norm_b = torch.linalg.vector_norm(g_b)
            if float(norm_a) < EPS or float(norm_b) < EPS:
                # Cosine against a zero vector is undefined, not zero: skip it.
                continue
            cosine = float((torch.dot(g_a, g_b) / (norm_a * norm_b)).item())
            self.cosine_history.setdefault((task_a, task_b), []).append(cosine)
            record["cosine"][f"{task_a}|{task_b}"] = cosine
            self._row(index, "cosine", task_a, task_b, cosine)

        record["triplet"] = self._finalize_triplet_stats(index)

        for task, count in acc.example_counts.items():
            self._row(index, "examples", task, "", count)
        self._row(index, "diag_secs", "", "", round(acc.diag_seconds, 4))
        self._row(index, "wall_secs", "", "", round(wall_seconds, 4))
        if torch.cuda.is_available():
            peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
            record["peak_mem_mib"] = round(peak_mib, 1)
            self._row(index, "peak_mem_mib", "", "", round(peak_mib, 1))
            torch.cuda.reset_peak_memory_stats()

        record["train_step"] = self.current_train_step
        self.virtual_batches.append(record)
        self._emit_virtual_batch(index, record, wall_seconds)

        # Reset accumulators for the next virtual batch.
        self._acc = _VirtualBatchAccumulator()

        if self.periodic and self.window_virtual_batches >= self.config.num_virtual_batches:
            self.emit_snapshot()

    def _finalize_triplet_stats(self, index: int) -> Dict[str, Any]:
        """Aggregate the virtual batch's per-microbatch triplet geometry."""
        acc = self._acc
        if not acc.triplet_stats:
            return {}

        populated = [s for s in acc.triplet_stats if s.get("num_triplets", 0) > 0]
        summary: Dict[str, Any] = {
            "microbatches": len(acc.triplet_stats),
            "empty_microbatches": acc.triplet_empty_microbatches,
            "total_triplets": sum(
                int(s.get("num_triplets", 0)) for s in acc.triplet_stats
            ),
        }
        # Averaged over microbatches that had triplets; the gradient still counts zero.
        for key in TRIPLET_STAT_KEYS:
            if key == "num_triplets":
                continue
            values = [float(s[key]) for s in populated if key in s]
            summary[key] = statistics.fmean(values) if values else None

        for key, value in summary.items():
            if value is None:
                continue
            self.triplet_history.setdefault(key, []).append(float(value))
            self._row(index, f"triplet_{key}", "triplet", "", value)
        return summary

    def _emit_virtual_batch(
        self, index: int, record: Dict[str, Any], wall_seconds: float
    ) -> None:
        examples = " ".join(
            f"{task}={count}" for task, count in sorted(record["examples"].items())
        )
        norms = " ".join(
            f"{task}={value:.4f}" for task, value in sorted(record["grad_norm"].items())
        )
        self._emit(
            f"[virtual batch {index}/{self.config.num_virtual_batches}] "
            f"microbatches={record['microbatches']} examples: {examples} "
            f"({wall_seconds:.1f}s wall, {record['diag_seconds']:.1f}s in autograd.grad)"
        )
        self._emit(f"    ||mean grad||: {norms}")
        triplet = record.get("triplet") or {}
        if triplet:
            active = triplet.get("active_fraction")
            self._emit(
                f"    triplets: {triplet['total_triplets']} over "
                f"{triplet['microbatches']} microbatches "
                f"({triplet['empty_microbatches']} empty), active="
                f"{'n/a' if active is None else f'{active:.2f}'}"
            )

    def emit_snapshot(self) -> Optional[Dict[str, Any]]:
        """Close the current measurement window and write its own file set.

        Returns the snapshot record, or None when nothing was measured (so a
        final snapshot at the end of training is a no-op if the last window
        happened to be empty).
        """
        if not self.virtual_batches:
            return None

        step = self.window_start_step
        snapshot: Dict[str, Any] = {
            "train_step": step,
            "last_train_step": self.current_train_step,
            "virtual_batches": list(self.virtual_batches),
            "num_virtual_batches": len(self.virtual_batches),
            "summary": self.summary(),
            "task_metrics": self.task_metrics(),
            "suggested_weights": self.suggested_weights(),
        }
        self.snapshots.append(snapshot)

        self._emit("")
        self._emit(
            f"--- snapshot @ step {step}: {len(self.virtual_batches)} virtual "
            f"batches -------------------"
        )
        for task in sorted(snapshot["summary"]["grad_norm"]):
            row = snapshot["summary"]["grad_norm"][task]
            loss = snapshot["summary"]["loss"].get(task, {}).get("median")
            self._emit(
                f"    {task:<10} ||g|| median {row['median']:>10.4f}   "
                f"loss {'n/a' if loss is None else f'{loss:10.4f}'}"
            )

        self._write_outputs(snapshot_step=step)
        self._reset_window()
        return snapshot

    def _reset_window(self) -> None:
        """Clear per-window state so the next snapshot stands alone; without it every
        snapshot pools all training so far and the trend flattens.
        """
        # The `_task_metrics` memo goes too, or the first window's numbers persist.
        self.norm_history = {}
        self.cosine_history = {}
        self.loss_history = {}
        self.triplet_history = {}
        self.virtual_batches = []
        self.window_virtual_batches = 0
        self._tone_preds = []
        self._tone_targets = []
        self._theme_logits = []
        self._theme_labels = []
        self._task_metrics = None

    def summary(self) -> Dict[str, Any]:
        """Summary statistics per task and per task pair (also used for JSON)."""
        norms = {task: describe(v) for task, v in self.norm_history.items()}
        cosines = {}
        for (task_a, task_b), values in self.cosine_history.items():
            stats = describe(values)
            # Plain fractions, no verdict: -0.05 half the time is orthogonality.
            stats["fraction_negative"] = sum(v < 0 for v in values) / len(values)
            stats["fraction_below_-0.3"] = sum(
                v < CONFLICT_COSINE for v in values
            ) / len(values)
            cosines[f"{task_a}|{task_b}"] = stats
        losses = {task: describe(v) for task, v in self.loss_history.items()}
        triplet = {key: describe(v) for key, v in self.triplet_history.items()}
        return {
            "grad_norm": norms,
            "cosine": cosines,
            "loss": losses,
            "triplet": triplet,
        }

    def task_metrics(self) -> Dict[str, Any]:
        """Tone / theme quality read-outs over the collected predictions, computed
        once and cached for both the text report and the JSON.
        """
        # Imported lazily so a missing scikit-learn/scipy does not stop the gradients.
        if self._task_metrics is not None:
            return self._task_metrics
        self._task_metrics = self._compute_task_metrics()
        return self._task_metrics

    def _compute_task_metrics(self) -> Dict[str, Any]:
        import numpy as np

        import task_metrics as tm

        metrics: Dict[str, Any] = {}
        if self._tone_preds:
            metrics["tone"] = tm.tone_metrics(self._tone_preds, self._tone_targets)
        if self._theme_logits:
            metrics["themes"] = tm.theme_metrics(
                np.concatenate(self._theme_logits, axis=0),
                np.concatenate(self._theme_labels, axis=0),
            )
        return metrics

    def suggested_weights(self) -> Dict[str, Any]:
        """Weights equalizing gradient magnitude against MLM: `w_i = sqrt(g_mlm / g_i)`
        over median norms. Informational only -- nothing writes these back.
        """
        # The sqrt is deliberate: a full g_mlm/g_i correction overshoots.
        medians = {task: statistics.median(v) for task, v in self.norm_history.items()}
        if REFERENCE_TASK not in medians:
            return {
                "reference": REFERENCE_TASK,
                "error": (
                    f"'{REFERENCE_TASK}' was not measured, so there is no reference "
                    "gradient to express the other tasks against."
                ),
                "median_grad_norm": medians,
            }

        reference = medians[REFERENCE_TASK]
        raw: Dict[str, float] = {}
        for task, median in medians.items():
            if task == REFERENCE_TASK:
                raw[task] = 1.0
            elif median <= EPS:
                raw[task] = float("nan")
            else:
                raw[task] = math.sqrt(reference / median)
        return {
            "reference": REFERENCE_TASK,
            "reference_median_grad_norm": reference,
            "formula": "w_i = sqrt(median||g_mlm|| / median||g_i||)",
            "clamp": [MIN_SUGGESTED_WEIGHT, MAX_SUGGESTED_WEIGHT],
            "rounding": f"floor onto a {WEIGHT_ROUNDING_GRID} grid",
            "median_grad_norm": medians,
            "raw": raw,
            "rounded": {task: conservative_weight(w) for task, w in raw.items()},
        }

    def report(self) -> None:
        # Open measurements become a final snapshot, so the run's end is represented.
        if self.periodic:
            self.emit_snapshot()
            self._report_trend()
            self._emit("=" * 78)
            self._write_outputs()
            return

        self._emit("")
        self._emit("=" * 78)
        self._emit(
            f"SUMMARY over {self.completed_virtual_batches} virtual batches of "
            f"{self.examples_per_virtual_batch} examples "
            f"({self.train_batch_size} x {self.config.microbatches_per_virtual_batch}), "
            f"scope: {self.config.parameter_scope}"
        )
        self._emit("=" * 78)

        if not self.norm_history:
            self._emit("No task gradients were collected.")
            self._write_outputs()
            return

        stats = self.summary()
        self._report_norms(stats)
        self._report_cosines(stats)
        self._report_triplet(stats)
        self._report_task_metrics()
        self._report_weights()
        self._emit("=" * 78)
        self._write_outputs()

    def _report_trend(self) -> None:
        """Per-snapshot tables: how each objective moved over training."""
        self._emit("")
        self._emit("=" * 78)
        self._emit(
            f"TREND over {len(self.snapshots)} snapshots "
            f"({self.config.num_virtual_batches} virtual batches each, "
            f"{self.examples_per_virtual_batch} examples per virtual batch), "
            f"scope: {self.config.parameter_scope}"
        )
        self._emit("=" * 78)

        if not self.snapshots:
            self._emit("No snapshots were collected.")
            return

        tasks = sorted(
            {task for s in self.snapshots for task in s["summary"]["grad_norm"]}
        )
        pairs = sorted({p for s in self.snapshots for p in s["summary"]["cosine"]})

        def table(title: str, note: str, columns: Sequence[str], cell) -> None:
            # Width follows the longest name, or pairs like "themes|triplet" collide.
            width = max(12, max((len(c) for c in columns), default=0) + 2)
            self._emit("")
            self._emit(title)
            self._emit("")
            header = f"{'step':>8}" + "".join(f"{c:>{width}}" for c in columns)
            self._emit(header)
            self._emit("-" * len(header))
            for snap in self.snapshots:
                row = f"{snap['train_step']:>8}"
                for column in columns:
                    value = cell(snap, column)
                    row += (
                        f"{'':>{width}}" if value is None
                        else f"{value:>{width}.4f}"
                    )
                self._emit(row)
            if note:
                self._emit("")
                self._emit(note)

        table(
            "Median gradient norm per task",
            "  Balance across tasks, not quality. Flat is fine; a task whose "
            "norm collapses toward 0 has stopped contributing.",
            tasks,
            lambda s, t: s["summary"]["grad_norm"].get(t, {}).get("median"),
        )
        table(
            "Median RAW (unweighted) task loss",
            "  THIS is the column that says whether a weighting is working: a "
            "task whose raw loss falls is learning, whatever its gradient norm "
            "does.",
            tasks,
            lambda s, t: s["summary"]["loss"].get(t, {}).get("median"),
        )
        if pairs:
            table(
                "Median gradient cosine per task pair",
                "  Watch for a pair drifting consistently below -0.3 as training "
                "proceeds; near-zero values are near-orthogonality, not conflict.",
                pairs,
                lambda s, p: s["summary"]["cosine"].get(p, {}).get("median"),
            )

        quality: List[Tuple[str, Any]] = [
            ("tone_pearson", lambda s: s["task_metrics"].get("tone", {}).get("pearson")),
            ("tone_spearman", lambda s: s["task_metrics"].get("tone", {}).get("spearman")),
            ("tone_rmse", lambda s: s["task_metrics"].get("tone", {}).get("rmse")),
            ("theme_f1_mic", lambda s: s["task_metrics"].get("themes", {}).get("f1_micro")),
            ("theme_ap_mic", lambda s: s["task_metrics"].get("themes", {}).get("average_precision_micro")),
        ]
        if any(fn(s) is not None for s in self.snapshots for _, fn in quality):
            lookup = dict(quality)
            table(
                "Task quality read-outs",
                "  Measured on the training batches themselves, so these say the "
                "heads are learning, not that they generalize (pretraining has no "
                "validation split). Still the most decision-relevant rows here.",
                [name for name, _ in quality],
                lambda s, c: lookup[c](s),
            )

        triplet_keys = ["active_fraction", "mean_violation", "empty_microbatches"]
        if any(
            s["summary"].get("triplet", {}).get(key) for s in self.snapshots
            for key in triplet_keys
        ):
            table(
                "Triplet geometry",
                "  A falling active_fraction means the model is solving the mined "
                "triplets; that shrinks the triplet gradient for a good reason, "
                "not a bad one.",
                triplet_keys,
                lambda s, c: s["summary"].get("triplet", {}).get(c, {}).get("median"),
            )

        self._emit("")
        self._emit("Suggested weights at the LAST snapshot (informational only)")
        self._emit("")
        last = self.snapshots[-1]["suggested_weights"]
        if "error" in last:
            self._emit(f"  {last['error']}")
        else:
            header = f"{'task':<12}{'gradient':>12}{'raw':>10}{'rounded':>10}"
            self._emit(header)
            self._emit("-" * len(header))
            for task in sorted(last["raw"], key=lambda t: -last["median_grad_norm"][t]):
                self._emit(
                    f"{task:<12}{last['median_grad_norm'][task]:>12.4f}"
                    f"{last['raw'][task]:>10.3f}{last['rounded'][task]:>10.2f}"
                )
        self._emit("")
        self._emit(
            "  Nothing here was written back into any config, and the weights "
            "this run used are unchanged. Choose a weighting from the raw-loss "
            "and quality trends above; the gradient norms only tell you whether "
            "the objectives are on comparable scales."
        )

    def _report_norms(self, stats: Dict[str, Any]) -> None:
        self._emit("")
        self._emit("Gradient norm summary  (||mean gradient|| per virtual batch)")
        self._emit("")
        header = (
            f"{'task':<12}{'median':>10}{'mean':>10}{'std':>10}"
            f"{'p5':>10}{'p95':>10}{'min':>10}{'max':>10}{'cv':>8}"
        )
        self._emit(header)
        self._emit("-" * len(header))
        for task in sorted(stats["grad_norm"]):
            row = stats["grad_norm"][task]
            self._emit(
                f"{task:<12}{row['median']:>10.4f}{row['mean']:>10.4f}"
                f"{row['std']:>10.4f}{row['p5']:>10.4f}{row['p95']:>10.4f}"
                f"{row['min']:>10.4f}{row['max']:>10.4f}{row['cv']:>8.2f}"
            )
        self._emit("")
        self._emit(
            "Median is the primary comparison. cv = std/|mean|: how unstable a "
            "task's gradient is from one virtual batch to the next, independent "
            "of how large it is."
        )

    def _report_cosines(self, stats: Dict[str, Any]) -> None:
        if not stats["cosine"]:
            return
        self._emit("")
        self._emit("Gradient cosine summary  (between AVERAGED task gradients)")
        self._emit("")
        header = (
            f"{'task A':<12}{'task B':<12}{'median':>9}{'mean':>9}"
            f"{'p5':>9}{'p95':>9}{'frac<0':>9}{'frac<-0.3':>11}"
        )
        self._emit(header)
        self._emit("-" * len(header))
        for pair in sorted(stats["cosine"]):
            task_a, task_b = pair.split("|")
            row = stats["cosine"][pair]
            self._emit(
                f"{task_a:<12}{task_b:<12}{row['median']:>+9.3f}{row['mean']:>+9.3f}"
                f"{row['p5']:>+9.3f}{row['p95']:>+9.3f}"
                f"{row['fraction_negative']:>9.2f}"
                f"{row['fraction_below_-0.3']:>11.2f}"
            )
        self._emit("")
        self._emit("Interpretation (conservative):")
        self._emit("  near 0            : largely unrelated gradients")
        self._emit("  positive          : objectives push in similar directions")
        self._emit("  consistently neg. : potential conflict")
        self._emit("  repeatedly <= -0.3: worth investigating")
        self._emit(
            "  A small negative median is NOT a harmful conflict. In a space this "
            "high-dimensional, independent gradients sit near orthogonal, and "
            "half of near-orthogonal pairs measure slightly negative."
        )

    def _report_triplet(self, stats: Dict[str, Any]) -> None:
        triplet = stats.get("triplet") or {}
        if not triplet:
            return
        self._emit("")
        self._emit("Triplet geometry  (per virtual batch, averaged over microbatches)")
        self._emit("")
        header = f"{'metric':<28}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}"
        self._emit(header)
        self._emit("-" * len(header))
        for key in sorted(triplet):
            row = triplet[key]
            self._emit(
                f"{key:<28}{row['median']:>10.4f}{row['mean']:>10.4f}"
                f"{row['min']:>10.4f}{row['max']:>10.4f}"
            )
        self._emit("")
        self._emit(
            "  active_fraction is the share of mined triplets with "
            "d_pos - d_neg + margin > 0, i.e. the ones that actually produce "
            "gradient. A low or swingy active fraction explains a high triplet "
            "gradient cv better than the loss value does."
        )
        self._emit(
            "  empty_microbatches counts local batches with no usable "
            "left/right split; each contributed a ZERO gradient vector to its "
            "virtual batch average (divided by K regardless)."
        )

    def _report_task_metrics(self) -> None:
        metrics = self.task_metrics()
        if not metrics:
            return

        import task_metrics as tm

        self._emit("")
        self._emit("Task quality read-outs  (untrained heads at this checkpoint)")
        self._emit("")
        if "tone" in metrics:
            for line in tm.tone_report_lines(metrics["tone"]):
                self._emit(line)
        if "themes" in metrics:
            for line in tm.theme_report_lines(metrics["themes"]):
                self._emit(line)
        self._emit("")
        self._emit(
            "  These score the heads as they stand, not the weighting. Do not "
            "pick loss weights from them -- and do not pick the theme weight "
            "from the magnitude of the theme loss either."
        )

    def _report_weights(self) -> None:
        weights = self.suggested_weights()
        self._emit("")
        self._emit("Suggested conservative weights (diagnostic suggestion only)")
        self._emit("")
        if "error" in weights:
            self._emit(f"  {weights['error']}")
            return

        header = f"{'task':<12}{'gradient':>12}{'raw suggestion':>18}{'rounded':>12}"
        self._emit(header)
        self._emit("-" * len(header))
        for task in sorted(weights["raw"], key=lambda t: -weights["median_grad_norm"][t]):
            self._emit(
                f"{task:<12}{weights['median_grad_norm'][task]:>12.4f}"
                f"{weights['raw'][task]:>18.3f}{weights['rounded'][task]:>12.2f}"
            )
        self._emit("")
        self._emit(
            f"  gradient = median ||mean gradient|| over "
            f"{self.completed_virtual_batches} virtual batches."
        )
        self._emit(
            f"  raw = sqrt(g_{REFERENCE_TASK} / g_task), with "
            f"w_{REFERENCE_TASK} = 1 by definition; rounded = clamped to "
            f"[{MIN_SUGGESTED_WEIGHT}, {MAX_SUGGESTED_WEIGHT}] and rounded down "
            f"onto a {WEIGHT_ROUNDING_GRID} grid."
        )
        self._emit(
            "  These are informational. No loss reweighting has been applied to "
            "the training objective, and nothing here is written back into any "
            "config -- copy a value across by hand only if you mean to."
        )
        self._emit("")
        self._emit("    loss_weights:")
        for task in ("triplet", "themes", "tone", "mlm"):
            value = weights["rounded"].get(task)
            suffix = "" if value is not None else "   # task not measured"
            self._emit(f"      {task}: {value if value is not None else 1.0}{suffix}")

    def _row(
        self, virtual_batch: int, metric: str, task_a: str, task_b: str, value: float
    ) -> None:
        self._rows.append(
            (
                self.run_id,
                self.run_label,
                self.config.parameter_scope,
                self.num_params,
                self.train_batch_size,
                self.config.microbatches_per_virtual_batch,
                self.examples_per_virtual_batch,
                self.checkpoint,
                self.current_train_step,
                self._loss_weights_tag(),
                virtual_batch,
                metric,
                task_a,
                task_b,
                value,
            )
        )

    def _loss_weights_tag(self) -> str:
        """Compact `triplet=1.0,themes=2.0,...` form for the TSV."""
        return ",".join(
            f"{task}={weight:g}"
            for task, weight in sorted(self.applied_loss_weights.items())
        )

    def _json_record(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "config": asdict(self.config),
            "provenance": {
                "timestamp": self.started_at.isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "seed": self.seed,
                "checkpoint": self.checkpoint,
                "argv": sys.argv,
            },
            "setup": {
                "train_batch_size": self.train_batch_size,
                "microbatches_per_virtual_batch": (
                    self.config.microbatches_per_virtual_batch
                ),
                "examples_per_virtual_batch": self.examples_per_virtual_batch,
                "target_global_batch_size": self.config.target_global_batch_size,
                "num_virtual_batches_requested": self.config.num_virtual_batches,
                "num_virtual_batches_completed": self.completed_virtual_batches,
                "parameter_scope": self.config.parameter_scope,
                "num_diag_params": self.num_params,
                "num_diag_tensors": len(self.parameters),
                "tracked_tasks": self.tracked_tasks,
                "loss_weights_in_effect": self.applied_loss_weights,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "virtual_batch_is_optimizer_step": self.virtual_batch_is_optimizer_step,
                "snapshot_interval_virtual_batches": (
                    self.config.diagnostics_every_n_virtual_batches
                ),
                "mixed_precision": getattr(self.accelerator, "mixed_precision", "no"),
                "num_processes": self.accelerator.num_processes,
                "device": str(getattr(self.accelerator, "device", "cpu")),
                **self.metadata,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "platform": platform.platform(),
            },
            "method": {
                "virtual_batch_gradient": (
                    "mean of K per-microbatch gradient vectors, K = "
                    "microbatches_per_virtual_batch; a microbatch with no loss "
                    "for a task contributes a zero vector and the sum is still "
                    "divided by K"
                ),
                "norm": "L2 norm of the mean gradient vector (not mean of norms)",
                "cosine": "between mean gradient vectors, skipped if either is zero",
                "cosine_epsilon": EPS,
                "gradients_via": "torch.autograd.grad(retain_graph=True, allow_unused=True)",
                "losses_reported": "raw, unweighted per-task losses",
            },
            "virtual_batches": self.virtual_batches,
            "summary": self.summary(),
            "task_metrics": self.task_metrics(),
            "suggested_weights": self.suggested_weights(),
            # Every snapshot so far, oldest first; empty for a one-shot run.
            "snapshots": self.snapshots,
        }

    def _write_outputs(self, snapshot_step: Optional[int] = None) -> None:
        """Write the artifacts; `snapshot_step` writes a self-contained
        `gradient_diagnostics_step-NNNNNN.{txt,json}` pair beside the cumulative TSV.
        """
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        if snapshot_step is not None:
            stem = f"gradient_diagnostics_step-{snapshot_step:06d}"
        else:
            stem = "gradient_diagnostics"

        txt_path = os.path.join(self.output_dir, f"{stem}.txt")
        with open(txt_path, "w") as handle:
            handle.write("\n".join(self._lines) + "\n")

        json_path = os.path.join(self.output_dir, f"{stem}.json")
        with open(json_path, "w") as handle:
            json.dump(self._json_record(), handle, indent=2)
            handle.write("\n")

        tsv_path = os.path.join(self.output_dir, "gradient_diagnostics.tsv")
        with open(tsv_path, "w") as handle:
            handle.write(TSV_HEADER_COMMENT)
            handle.write("\t".join(TSV_COLUMNS) + "\n")
            for row in self._rows:
                handle.write("\t".join(str(v) for v in row) + "\n")

        written = [txt_path, json_path, tsv_path]

        # Shared sweep file: append only new rows, or snapshots re-append history.
        if self.config.records_path and len(self._rows) > self._rows_flushed:
            shared = self.config.records_path
            os.makedirs(os.path.dirname(shared) or ".", exist_ok=True)
            exists = os.path.exists(shared)
            with open(shared, "a") as handle:
                if not exists:
                    handle.write(TSV_HEADER_COMMENT)
                    handle.write("\t".join(TSV_COLUMNS) + "\n")
                for row in self._rows[self._rows_flushed :]:
                    handle.write("\t".join(str(v) for v in row) + "\n")
            written.append(f"{shared} (appended)")
        self._rows_flushed = len(self._rows)

        print("\nGradient diagnostics written to:")
        for path in written:
            print(f"   {path}")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._emit(message)

    def _emit(self, line: str) -> None:
        print(line)
        self._lines.append(line)
