import argparse
import torch
import torch.nn as nn
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
from dataset import build_dataloaders
from gradient_diagnostics import GradientDiagnostics
from model import MultiTaskRoberta
from pretraining_utils import calculate_eta, load_run_config
from transformers.models.auto.tokenization_auto import AutoTokenizer
import time
from transformers.optimization import get_linear_schedule_with_warmup
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to run YAML config")
cli = parser.parse_args()
cfg = load_run_config(cli.config)

output_dir = cfg.output_dir
os.makedirs(output_dir, exist_ok=True)

# Offset epoch numbering when continuing from an epoch-N.pt checkpoint, so we
# don't overwrite the source file when output_dir is shared.
epoch_offset = 0
if cfg.init_from_checkpoint:
    m = re.match(r"epoch-(\d+)", os.path.basename(cfg.init_from_checkpoint))
    if m:
        epoch_offset = int(m.group(1))

# Set random seed
set_seed(42)

# --- Training prep ---
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# Long timeout so rank-0-only setup (filter-index build, dataset download) doesn't
# trip the default 10-min NCCL barrier on cold caches.
pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    project_dir=output_dir,
    kwargs_handlers=[ddp_kwargs, pg_kwargs],
)

args = cfg.train_args

# Temporary gradient diagnostic (see gradient_diagnostics.py). Fail fast, before
# the dataset is loaded, if it was launched on more than one process: the
# per-task torch.autograd.grad calls must not run on a DDP-wrapped model.
diag_cfg = cfg.gradient_diagnostics
if diag_cfg.enabled and accelerator.num_processes != 1:
    raise RuntimeError(
        "Gradient diagnostics must be run with a single Accelerate process."
    )
diagnostic_only = diag_cfg.enabled and diag_cfg.diagnostic_only

# Define which tasks to run
available_tasks = cfg.tasks


# Multi-GPU information
if accelerator.is_main_process:
    print("Training Setup:")
    print(f"   - Number of processes: {accelerator.num_processes}")
    print(f"   - Distributed type: {accelerator.distributed_type}")
    print(f"   - Mixed precision: {accelerator.mixed_precision}")
    print(f"   - Main process: {accelerator.is_main_process}")
    if torch.cuda.is_available():
        print(f"   - CUDA devices: {torch.cuda.device_count()}")
        print(f"   - Current device: {accelerator.device}")

print(f"TrainArgs loaded: {args.num_epochs} epochs, {args.model_name}")

theme_count = cfg.theme_count
subsample = cfg.task_spec.require_nonempty_themes_and_tone

ckpt = None
ckpt_num_themes = theme_count
ckpt_num_tones = 1
ckpt_num_bias_classes = None
if cfg.init_from_checkpoint:
    if accelerator.is_main_process:
        print(f"Loading initial weights from {cfg.init_from_checkpoint}")
    ckpt = torch.load(cfg.init_from_checkpoint, map_location="cpu", weights_only=False)
    # Prefer head sizes baked into the checkpoint over current config defaults,
    # otherwise load_state_dict will fail on shape mismatch.
    ckpt_config = ckpt.get("config", {}) or {}
    state = ckpt["model_state_dict"]
    ckpt_num_themes = ckpt_config.get(
        "num_themes", state["theme_head.weight"].shape[0]
    )
    ckpt_num_tones = ckpt_config.get(
        "num_tones", state["tone_head.weight"].shape[0]
    )
    ckpt_num_bias_classes = ckpt_config.get("num_bias_classes", None)
    if accelerator.is_main_process:
        print(
            f"  Using checkpoint head sizes: num_themes={ckpt_num_themes}, "
            f"num_tones={ckpt_num_tones}, num_bias_classes={ckpt_num_bias_classes}"
        )

loss_weights = cfg.loss_weights.as_dict()
if accelerator.is_main_process:
    label = "defaults (unweighted sum)" if cfg.loss_weights.is_default() else "OVERRIDDEN"
    print(f"Per-task loss weights [{label}]: {loss_weights}")

model = MultiTaskRoberta(
    num_themes=ckpt_num_themes,
    num_tones=ckpt_num_tones,
    num_bias_classes=ckpt_num_bias_classes,
    loss_weights=loss_weights,
)

if ckpt is not None:
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if accelerator.is_main_process:
        if missing:
            print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        print("  Checkpoint loaded")

model.to(accelerator.device)
print(f"Model initialized: {model.__class__.__name__}")

# Reduce activation memory: re-compute intermediate activations during backward
# instead of storing them. Trades ~20-30% step time for ~5-10x lower activation
# memory. Required for batch_size=32 with the triplet task active.
# Uses non-reentrant checkpointing (see model.gradient_checkpointing_enable)
# because reentrant checkpointing trips DDP's "marked ready twice" assertion
# when the shared backbone is forwarded multiple times per step.
# model.backbone.config.use_cache = False
# model.gradient_checkpointing_enable()
# if accelerator.is_main_process:
#     print("Gradient checkpointing enabled on backbone (use_reentrant=False)")

effective_batch_size = (
    args.batch_size
    * accelerator.num_processes
    * accelerator.gradient_accumulation_steps
)
base_lr = cfg.base_lr
optimizer = torch.optim.AdamW(
    model.parameters(), lr=base_lr, weight_decay=0.01, fused=True
)
print("Optimizer created")

print("\nBuilding dataloaders for multi-task training...")
print(" - Loading tokenizer for roberta-base...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("   Tokenizer loaded")

task_spec = cfg.task_spec

print(" - Building dataloaders (this may take a moment)...")
dataloaders = build_dataloaders(
    tok=tokenizer, task_spec=task_spec, args=args, tasks_to_build=available_tasks
)
print("   Dataloaders built")

# Calculate dataset sizes and steps per epoch
dataset_sizes = {}
steps_per_epoch = {}

for task_name, dataloader in dataloaders.items():
    if dataloader is not None:
        dataset_sizes[task_name] = len(dataloader.dataset)
        steps_per_epoch[task_name] = len(dataloader)
        print(
            f"   {task_name:12s}: {dataset_sizes[task_name]:,} samples, {steps_per_epoch[task_name]:,} steps per epoch"
        )
    else:
        print(f"   {task_name:12s}: No dataloader created (skipped)")

# Calculate total training steps based on epochs and largest dataset
max_steps_per_epoch = (
    max(steps_per_epoch.values()) // accelerator.num_processes
    if steps_per_epoch
    else 1000
)
TOTAL_STEPS = args.num_epochs * max_steps_per_epoch
print("\n Training configuration:")
print(f"   - Epochs: {args.num_epochs}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Total training steps: {TOTAL_STEPS:,}")

print("\n Creating learning rate scheduler...")
print(f"   - Warmup ratio: {args.warmup_ratio}")
print(f"   - Warmup steps: {int(TOTAL_STEPS * args.warmup_ratio):,}")
# Add learning rate scheduler

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(TOTAL_STEPS * args.warmup_ratio),
    num_training_steps=TOTAL_STEPS,
)

print("\nPreparing objects with Accelerator...")
# Prepare dataloaders for distributed training as well.
prepared_dataloaders = {
    name: (accelerator.prepare(dl) if dl is not None else None)
    for name, dl in dataloaders.items()
}

(model, scheduler, optimizer) = accelerator.prepare(model, scheduler, optimizer)
dataloaders = prepared_dataloaders  # Use prepared dataloaders

if accelerator.is_main_process:
    print("Model, optimizer, scheduler, and dataloaders prepared with Accelerator")
    print(f"   - Model device: {next(model.parameters()).device}")
    print(f"   - Effective batch size per GPU: {args.batch_size}")
    print(
        f"   - Total effective batch size: {args.batch_size * accelerator.num_processes}"
    )

# Initialize native logging configuration
if accelerator.is_main_process:
    print("\nTraining Configuration Summary:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Learning rate: {base_lr}")
    print("   - Weight decay: 0.01")
    print("   - Mixed precision: fp16")
    print(f"   - Number of GPUs: {accelerator.num_processes}")
    print(
        f"   - Gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    print(
        f"   - Effective batch sizes : {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}"
    )
    print(f"   - Dataset: {task_spec.dataset_name}")
    print(f"   - Total training steps: {TOTAL_STEPS:,}")

print("\n Initializing loss functions...")
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
print(" Loss functions initialized:")
print("   - Triplet loss (margin=1.0)")
print("   - BCE with logits loss (for themes)")
print("   - MSE loss (for tone regression)")

print("\n Initializing task iterators and training order...")
iters = {}

# Maps logical task name -> dataloader key
TASK_TO_DL = {
    "triplet": "triplet",
    "mlm": "mlm",
    "tone": "tone",
    "themes": "themes",
}

for task_name, dl_name in TASK_TO_DL.items():
    if dataloaders.get(dl_name):
        iters[task_name] = iter(dataloaders[dl_name])

print("\nTraining configuration complete:")
print(f"   - Available tasks: {available_tasks}")
print(f"   - Tasks per cycle: {len(available_tasks)}")

if not available_tasks:
    print("ERROR: No tasks available for training!")
    exit(1)

# Create log file. In diagnostic-only mode we do not train, so don't truncate an
# existing training log; the diagnostic writes its own file instead.
log_file = f"./{output_dir}/training_log.txt"
if diagnostic_only:
    print("   Diagnostic-only run: leaving training_log.txt untouched")
else:
    with open(log_file, "w") as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Start Time: {__import__('datetime').datetime.now()}\n")
        f.write(f"Model: {args.model_name}\n")
        if cfg.init_from_checkpoint:
            f.write(f"Init from checkpoint: {cfg.init_from_checkpoint}\n")
        f.write(f"Dataset: {task_spec.dataset_name}\n")
        f.write(f"Total Steps: {TOTAL_STEPS:,}\n")
        f.write(f"GPUs: {accelerator.num_processes}\n")
        f.write(f"Available Tasks: {available_tasks}\n")
        f.write(f"Using num themes: {theme_count} and num tones: 1\n")
        f.write(f"Num triplets per batch: {task_spec.max_triplet_samples}\n")
        f.write(f"Subsampling for themes/tone: {subsample}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Num epochs: {args.num_epochs}\n")
        f.write(f"Loss weights: {loss_weights}\n")
        f.write("=" * 50 + "\n\n")
    print(f"   Logging initialized: {log_file}")

# Build the diagnostic after the model is prepared, so it can unwrap the model
# and hold references to the shared-representation parameters.
diagnostics = None
if diag_cfg.enabled:
    diagnostics = GradientDiagnostics(
        accelerator=accelerator,
        model=model,
        config=diag_cfg,
        output_dir=output_dir,
        train_batch_size=args.batch_size,
        # Recorded verbatim in the JSON/TXT so a reader can reproduce the run.
        metadata={
            "config_path": cli.config,
            "seed": 42,
            "model_name": args.model_name,
            "dataset": task_spec.dataset_name,
            "tasks": ",".join(available_tasks),
            "max_triplet_samples": task_spec.max_triplet_samples,
            "require_nonempty_themes_and_tone": subsample,
            "init_from_checkpoint": cfg.init_from_checkpoint or "none",
            "base_lr": base_lr,
        },
    )

step = 0
epoch = 0
steps_in_current_epoch = 0
diagnostics_reported = False
model.train()

# Loss tracking
loss_accumulator = {
    "triplet": [],
    "mlm": [],
    "tone": [],
    "themes": [],
}

# Missing samples tracking
missing_samples = {
    "triplet": 0,
    "mlm": 0,
    "tone": 0,
    "themes": 0,
}

print(f"\n Loss tracking initialized for {len(loss_accumulator)} tasks")


def get_next_batch(task_name, iterators, dataloaders):
    """Get next batch, reinitializing iterator if exhausted."""
    try:
        return next(iterators[task_name])
    except StopIteration:
        print(f"   Reinitializing {task_name} iterator (end of dataset reached)")
        dl_name = TASK_TO_DL.get(task_name, task_name)
        iterators[task_name] = iter(dataloaders[dl_name])
        return next(iterators[task_name])


# Add timing variables for ETC calculation
training_start_time = time.time()
step_times = []  # Store recent step times for averaging
eta_window_size = 50  # Number of recent steps to average for ETC
avg_step_time = 0.0  # Initialize avg_step_time

print("\n" + "=" * 80)
print(f"STARTING TRAINING: {args.num_epochs} EPOCHS")
print("=" * 80)

# torch.autograd.set_detect_anomaly(True)

while epoch < args.num_epochs:
    epoch_start_step = step
    epoch_start_time = time.time()  # Track epoch start time
    print(f"\nEPOCH {epoch + 1}/{args.num_epochs}")
    print(f"   Target steps this epoch: {max_steps_per_epoch:,}")
    print("-" * 50)

    while steps_in_current_epoch < max_steps_per_epoch and step < TOTAL_STEPS:
        with accelerator.accumulate(model):
            # 1. Combine batches from all available tasks into one
            combined_batch = {}
            for task_name in available_tasks:
                dl_name = TASK_TO_DL.get(task_name)
                if not dl_name:
                    continue

                batch = get_next_batch(task_name, iters, dataloaders)
                if batch is not None and not batch.get("_skip", False):
                    # Rename keys to be unique and descriptive for the model
                    if task_name == "triplet":
                        combined_batch.update(
                            {
                                "triplet_a_ids": batch["a_ids"],
                                "triplet_a_mask": batch["a_mask"],
                                "triplet_p_ids": batch["p_ids"],
                                "triplet_p_mask": batch["p_mask"],
                                "triplet_n_ids": batch["n_ids"],
                                "triplet_n_mask": batch["n_mask"],
                            }
                        )
                    elif task_name == "themes":
                        combined_batch.update(
                            {
                                "theme_input_ids": batch["input_ids"],
                                "theme_attention_mask": batch["attention_mask"],
                                "theme_labels": batch["labels"],
                            }
                        )
                    elif task_name == "tone":
                        combined_batch.update(
                            {
                                "tone_input_ids": batch.get("input_ids"),
                                "tone_attention_mask": batch.get("attention_mask"),
                                "tone_labels": batch["targets"],
                            }
                        )
                    elif task_name == "mlm":
                        combined_batch.update(
                            {
                                "mlm_input_ids": batch["input_ids"],
                                "mlm_attention_mask": batch["attention_mask"],
                                "mlm_labels": batch["labels"],
                            }
                        )

            # If the combined batch is empty, skip this step
            if not combined_batch:
                step += 1
                steps_in_current_epoch += 1
                continue

            # 2. Perform a single forward pass with the combined batch
            outputs = model(**combined_batch)
            total_loss = outputs.get("loss")

            # 2b. Diagnostic (optional): per-task gradients on the shared
            # representation. Runs after the forward pass and BEFORE the normal
            # backward; uses torch.autograd.grad(retain_graph=True), so it never
            # writes to .grad and leaves the graph intact for step 3.
            if diagnostics is not None:
                diagnostics.record_microbatch(outputs, combined_batch)

                if diagnostic_only:
                    # No backward, no optimizer/scheduler step: every virtual
                    # batch measures the same model state. Drop the graph before
                    # the next forward pass so memory doesn't stack up.
                    del outputs, total_loss, combined_batch
                    step += 1
                    steps_in_current_epoch += 1
                    if diagnostics.is_complete():
                        break
                    continue

                if diagnostics.is_complete() and not diagnostics_reported:
                    # Diagnostics ran alongside real training; report once and
                    # let training carry on untouched.
                    diagnostics.report()
                    diagnostics_reported = True

            # 3. Perform a single backward pass on the combined loss
            if total_loss is not None:
                accelerator.backward(total_loss)

                # Log individual losses returned from the model
                if accelerator.is_main_process:
                    if (
                        "triplet_loss" in outputs
                        and outputs["triplet_loss"] is not None
                    ):
                        loss_accumulator["triplet"].append(
                            outputs["triplet_loss"].item()
                        )
                    if "theme_loss" in outputs and outputs["theme_loss"] is not None:
                        loss_accumulator["themes"].append(outputs["theme_loss"].item())
                    if "tone_loss" in outputs and outputs["tone_loss"] is not None:
                        loss_accumulator["tone"].append(outputs["tone_loss"].item())
                    if "mlm_loss" in outputs and outputs["mlm_loss"] is not None:
                        loss_accumulator["mlm"].append(outputs["mlm_loss"].item())

            # 4. Gradient clipping and optimizer step
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and step % args.log_every == 0:
            # Calculate average losses for each task
            task_losses = {}
            for task_name in ["triplet", "mlm", "tone", "themes"]:
                if loss_accumulator[task_name]:
                    avg_loss = sum(loss_accumulator[task_name]) / len(
                        loss_accumulator[task_name]
                    )
                    task_losses[task_name] = avg_loss
                    loss_accumulator[task_name] = []  # Reset accumulator

            if task_losses:  # Only log if there were any losses in this interval
                current_lr = scheduler.get_last_lr()[0]

                # Timing calculations
                (
                    eta_str,
                    completion_str,
                    elapsed_str,
                    epoch_eta_str,
                    progress_pct,
                    epoch_progress_pct,
                    steps_per_second,
                    samples_per_second,
                    avg_step_time,
                ) = calculate_eta(
                    training_start_time,
                    epoch_start_time,
                    step_times,
                    step,
                    steps_in_current_epoch,
                    max_steps_per_epoch,
                    TOTAL_STEPS,
                    avg_step_time,
                    args.batch_size,
                    accelerator.num_processes,
                )

                # Prepare logging data
                log_entry = f"Step {step:5d} | LR: {current_lr:.2e} | Epoch: {epoch + 1} | Progress: {epoch_progress_pct:.1f}%"

                print(f"  {log_entry}")
                print(
                    f"    Elapsed: {elapsed_str} | Epoch {epoch + 1}: {epoch_progress_pct:.1f}% | Epoch ETC: {epoch_eta_str}"
                )
                if task_losses:
                    task_loss_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in task_losses.items()]
                    )
                    print(f"    Avg Task Losses: {task_loss_str}")

                # Log to file
                with open(log_file, "a") as f:
                    f.write(f"{log_entry}\n")
                    f.write(
                        f"  Timing: Elapsed={elapsed_str}, Epoch: {epoch + 1} ({epoch_progress_pct:.1f}%), Epoch ETC: {epoch_eta_str}\n"
                    )
                    f.write("\n")

        step += 1
        steps_in_current_epoch += 1

    if diagnostic_only:
        # Diagnostic-only run: no checkpoint, no epoch bookkeeping.
        break

    # End of epoch
    epoch_steps = step - epoch_start_step
    print(f"\nCOMPLETED EPOCH {epoch + 1}")
    print(f"   Steps completed: {epoch_steps:,}")
    print(f"   Total steps so far: {step:,}")

    # Save checkpoint at end of epoch
    if accelerator.is_main_process:
        epoch_checkpoint_path = f"{output_dir}/epoch-{epoch_offset + epoch + 1}"
        print(f"Saving epoch checkpoint to {epoch_checkpoint_path}...")
        accelerator.unwrap_model(model).save_checkpoint(f"{epoch_checkpoint_path}.pt")
        print("Epoch checkpoint saved")

    # Reset epoch counter and reinitialize all iterators for next epoch
    steps_in_current_epoch = 0
    epoch += 1

    if epoch < args.num_epochs:  # Don't reinitialize on the last epoch
        print("Preparing for next epoch...")
        # Reinitialize iterators for next epoch
        for task_name in iters.keys():
            dl_name = TASK_TO_DL.get(task_name, task_name)
            if dl_name and dataloaders.get(dl_name):
                iters[task_name] = iter(dataloaders[dl_name])
        print(f"All iterators reinitialized for epoch {epoch + 1}")

    print("=" * 50)

if diagnostic_only and diagnostics is not None:
    diagnostics.report()
    print(
        f"\nDiagnostic-only run: {step} forward passes, no optimizer steps taken. "
        "Training objective and weights are unchanged."
    )
    accelerator.end_training()
    raise SystemExit(0)

# Clean shutdown
print("\nTraining completed! Cleaning up...")
accelerator.end_training()
print("Accelerator shutdown complete")

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("FINAL STATISTICS:")
print(f"   - Total epochs completed: {args.num_epochs}")
print(f"   - Total steps executed: {step:,}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Tasks trained: {', '.join(available_tasks)}")
print("=" * 80)
