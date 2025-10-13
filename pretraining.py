import torch, torch.nn as nn, torch.nn.functional as F
from accelerate import Accelerator
from dataset import build_dataloaders
from model import MultiTaskRoberta
from pretraining_utils import TrainArgs, TaskSpec, update_loss, calculate_eta
from transformers.models.auto.tokenization_auto import AutoTokenizer
import time
from transformers.optimization import get_cosine_schedule_with_warmup
import os

# Environment setup (wandb removed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Create output directories
output_dir = "./mtl_ckpt"
os.makedirs(output_dir, exist_ok=True)

# --- Training prep ---
accelerator = Accelerator(
    mixed_precision="fp16", gradient_accumulation_steps=8, project_dir=output_dir
)

args = TrainArgs()


# Multi-GPU information
if accelerator.is_main_process:
    print(f"Training Setup:")
    print(f"   - Number of processes: {accelerator.num_processes}")
    print(f"   - Distributed type: {accelerator.distributed_type}")
    print(f"   - Mixed precision: {accelerator.mixed_precision}")
    print(f"   - Main process: {accelerator.is_main_process}")
    if torch.cuda.is_available():
        print(f"   - CUDA devices: {torch.cuda.device_count()}")
        print(f"   - Current device: {accelerator.device}")

print(f"TrainArgs loaded: {args.num_epochs} epochs, {args.model_name}")

# Check if themes file exists
import os

themes_path = "top_themes.txt"
theme_count = 0

with open(themes_path, "r") as f:
    theme_count = len(f.readlines())

model = MultiTaskRoberta(theme_path=themes_path)
model.to(accelerator.device)
print(f"Model initialized: {model.__class__.__name__}")

effective_batch_size = (
    args.batch_size
    * accelerator.num_processes
    * accelerator.gradient_accumulation_steps
)
base_lr = 5e-5  # Start with a more conservative base learning rate
adjusted_lr = base_lr * (
    effective_batch_size / 32
)  # Scale based on effective batch size
optimizer = torch.optim.AdamW(model.parameters(), lr=adjusted_lr, weight_decay=0.01, fused=True)
print("Optimizer created")

print(f"\nBuilding dataloaders for multi-task training...")
print(" - Loading tokenizer for roberta-base...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("   Tokenizer loaded")

task_spec = TaskSpec(
    dataset_name="dragonslayer631/bignewsalign-with-gdelt", themes_path="top_themes.txt"
)

print(" - Building dataloaders (this may take a moment)...")
dataloaders = build_dataloaders(tok=tokenizer, task_spec=task_spec, args=args)
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
max_steps_per_epoch = max(steps_per_epoch.values()) if steps_per_epoch else 1000
TOTAL_STEPS = args.num_epochs * max_steps_per_epoch
print(f"\n Training configuration:")
print(f"   - Epochs: {args.num_epochs}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Total training steps: {TOTAL_STEPS:,}")

print(f"\n Creating learning rate scheduler...")
print(f"   - Warmup ratio: {args.warmup_ratio}")
print(f"   - Warmup steps: {int(TOTAL_STEPS * args.warmup_ratio):,}")
# Add learning rate scheduler

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(TOTAL_STEPS * args.warmup_ratio),
    num_training_steps=TOTAL_STEPS,
    num_cycles=0.5,  # This creates a gentle decay
)

print(f"\nPreparing objects with Accelerator...")
# Prepare dataloaders for distributed training as well
prepared_dataloaders = {}
for name, dl in dataloaders.items():
    if dl is not None:
        prepared_dataloaders[name] = accelerator.prepare(dl)
    else:
        prepared_dataloaders[name] = None

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
    print(f"\nTraining Configuration Summary:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Learning rate: 0.0005")
    print(f"   - Weight decay: 0.01")
    print(f"   - Mixed precision: fp16")
    print(f"   - Number of GPUs: {accelerator.num_processes}")
    print(
        f"   - Gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    print(
        f"   - Effective batch sizes : {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}"
    )
    print(f"   - Dataset: {task_spec.dataset_name}")
    print(f"   - Total training steps: {TOTAL_STEPS:,}")

print(f"\n Initializing loss functions...")
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
print(" Loss functions initialized:")
print("   - Triplet loss (margin=1.0)")
print("   - BCE with logits loss (for themes)")
print("   - MSE loss (for tone regression)")

print(f"\n Initializing task iterators and training order...")
iters = {}
available_tasks = []

# Initialize iterators for available dataloaders
if dataloaders.get("triplet"):
    iters["triplet"] = iter(dataloaders["triplet"])
    available_tasks.append("triplet")

if dataloaders.get("mlm"):
    iters["mlm"] = iter(dataloaders["mlm"])
    available_tasks.append("mlm")

if dataloaders.get("regression"):
    iters["tone"] = iter(dataloaders["regression"])
    available_tasks.append("tone")

if dataloaders.get("multilabel"):
    iters["themes"] = iter(dataloaders["multilabel"])
    available_tasks.append("themes")

# Build training order based on available tasks
order = []
if "triplet" in available_tasks:
    order.extend(["triplet", "triplet"])  # Triplet appears twice as in original
if "mlm" in available_tasks:
    order.append("mlm")
if "tone" in available_tasks:
    order.insert(-1 if order else 0, "tone")
if "themes" in available_tasks:
    order.insert(-1 if order else 0, "themes")


## manually setting order for testing
order = ["no more round robin -- using all tasks every step"]
available_tasks = ["triplet", "mlm"]

for task, label in zip(
    ["triplet", "mlm", "themes", "tone"], ["triplet", "mlm", "multilabel", "regression"]
):
    if task not in available_tasks:
        print(f"removing dataloader for task: {task}")
        del dataloaders[label]

print(f"\nTraining configuration complete:")
print(f"   - Training order: {order}")
print(f"   - Available tasks: {available_tasks}")
print(f"   - Tasks per cycle: {len(available_tasks)}")

if not available_tasks:
    print("ERROR: No tasks available for training!")
    exit(1)

# Create log file
log_file = f"./{output_dir}/training_log.txt"
with open(log_file, "w") as f:
    f.write("Training Log\n")
    f.write("=" * 50 + "\n")
    f.write(f"Start Time: {__import__('datetime').datetime.now()}\n")
    f.write(f"Model: {args.model_name}\n")
    f.write(f"Dataset: {task_spec.dataset_name}\n")
    f.write(f"Total Steps: {TOTAL_STEPS:,}\n")
    f.write(f"GPUs: {accelerator.num_processes}\n")
    f.write(f"Order: {order}\n")
    f.write(f"Available Tasks: {available_tasks}\n")
    f.write("=" * 50 + "\n\n")
print(f"   Logging initialized: {log_file}")

step = 0
epoch = 0
steps_in_current_epoch = 0
model.train()

# Loss tracking
loss_accumulator = {"triplet": [], "mlm": [], "tone": [], "themes": []}

print(f"\n Loss tracking initialized for {len(loss_accumulator)} tasks")


def get_next_batch(task_name, iterators, dataloaders):
    """Get next batch, reinitializing iterator if exhausted."""
    try:
        return next(iterators[task_name])
    except StopIteration:
        print(f"   Reinitializing {task_name} iterator (end of dataset reached)")
        # Map task names to dataloader names
        dataloader_mapping = {
            "triplet": "triplet",
            "mlm": "mlm",
            "tone": "regression",
            "themes": "multilabel",
        }
        dl_name = dataloader_mapping.get(task_name, task_name)
        iterators[task_name] = iter(dataloaders[dl_name])
        return next(iterators[task_name])


# Add timing variables for ETC calculation
training_start_time = time.time()
step_times = []  # Store recent step times for averaging
eta_window_size = 50  # Number of recent steps to average for ETC
avg_step_time = 0.0  # Initialize avg_step_time

print(f"\n" + "=" * 80)
print(f"STARTING TRAINING: {args.num_epochs} EPOCHS")
print(f"=" * 80)

while epoch < args.num_epochs:
    epoch_start_step = step
    epoch_start_time = time.time()  # Track epoch start time
    print(f"\nEPOCH {epoch + 1}/{args.num_epochs}")
    print(f"   Target steps this epoch: {max_steps_per_epoch:,}")
    print("-" * 50)

    while steps_in_current_epoch < max_steps_per_epoch and step < TOTAL_STEPS:
        step_start_time = time.time()  # Track individual step time

        loss = None  # Initialize loss variable

        # Use gradient accumulation context for the entire step
        with accelerator.accumulate(model):
            # Forward pass (no autocast needed when mixed precision is disabled)
            if "triplet" in available_tasks:
                b = get_next_batch("triplet", iters, dataloaders)
                if b is None or b.get("_skip", False):
                    # Skip this step if no valid triplet batch could be formed - use dummy loss
                    print(
                        f"   Skipping triplet step {step} - no valid triplets found, using dummy loss"
                    )
                    loss_tlp = torch.tensor(0.0, device=accelerator.device)
                else:
                    za, _ = model(
                        b["a_ids"].to(accelerator.device).contiguous().clone(),
                        b["a_mask"].to(accelerator.device).contiguous().clone()
                    )
                    zp, _ = model(
                        b["p_ids"].to(accelerator.device).contiguous().clone(),
                        b["p_mask"].to(accelerator.device).contiguous().clone()
                    )
                    zn, _ = model(
                        b["n_ids"].to(accelerator.device).contiguous().clone(),
                        b["n_mask"].to(accelerator.device).contiguous().clone()
                    )
                    loss_tlp = triplet_loss(za, zp, zn)
                    if loss_tlp is not None:
                        loss_accumulator["triplet"].append(loss_tlp.item())

                loss = update_loss(loss_tlp, loss)

            if "themes" in available_tasks:
                b = get_next_batch("themes", iters, dataloaders)
                if b is None or b.get("_skip", False):
                    # Skip this step if no valid themes batch - use dummy loss
                    print(
                        f"     Skipping themes step {step} - no valid samples, using dummy loss"
                    )
                    loss_themes = torch.tensor(0.0, device=accelerator.device)
                else:
                    z, pooled = model(b["input_ids"], b["attention_mask"])
                    logits = model.theme_head(pooled)
                    loss_themes = bce_loss(logits, b["labels"].float())
                    if loss_themes is not None:
                        loss_accumulator["themes"].append(loss_themes.item())

                loss = update_loss(loss_themes, loss)

            if "tone" in available_tasks:
                b = get_next_batch("tone", iters, dataloaders)
                if b is None or b.get("_skip", False):
                    # Skip this step if no valid tone batch - use dummy loss
                    print(
                        f"     Skipping tone step {step} - no valid samples, using dummy loss"
                    )
                    loss_tone = torch.tensor(0.0, device=accelerator.device)
                else:
                    z, pooled = model(b["input_ids"], b["attention_mask"])
                    pred = model.tone_head(pooled)
                    loss_tone = mse_loss(pred, b["targets"].float())
                    if loss_tone is not None:
                        loss_accumulator["tone"].append(loss_tone.item())

                loss = update_loss(loss_tone, loss)

            if "mlm" in available_tasks:
                b = get_next_batch("mlm", iters, dataloaders)
                out = model(mlm=True, **b)
                loss_mlm = out.loss
                if loss_mlm is not None:
                    loss_accumulator["mlm"].append(loss_mlm.item())

                loss = update_loss(loss_mlm, loss)

            # Only proceed with backpropagation if loss is not None
            if loss is not None:
                accelerator.backward(loss)

                # Gradient clipping and optimizer step - let accelerator handle sync_gradients
                if accelerator.sync_gradients:
                    # Clip gradients using accelerator's method which handles mixed precision properly
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                print(f"Warning: Loss is None at step {step}, skipping backward pass")

        if accelerator.is_main_process and step % args.log_every == 0:
            if loss is not None:
                # Calculate average losses for each task
                current_loss = float(loss.item())
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

                # Add task-specific losses if they have data
                task_losses = {}
                for task_name in ["triplet", "mlm", "tone", "themes"]:
                    if loss_accumulator[task_name]:
                        avg_loss = sum(loss_accumulator[task_name]) / len(
                            loss_accumulator[task_name]
                        )
                        task_losses[task_name] = avg_loss
                        loss_accumulator[task_name] = []  # Reset accumulator

                # Log to console
                # Log to console with rich timing info
                print(f"  {log_entry}")
                print(
                    f"    Elapsed: {elapsed_str} | ETC: {eta_str} | Completion: {completion_str}"
                )
                print(
                    f"    Speed: {steps_per_second:.2f} steps/s | {samples_per_second:.0f} samples/s"
                )
                print(
                    f"    Epoch {epoch + 1}: {epoch_progress_pct:.1f}% | Epoch ETC: {epoch_eta_str}"
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
                        f"  Timing: Elapsed={elapsed_str}, ETC={eta_str}, Completion={completion_str}\n"
                    )
                    f.write(
                        f"  Speed: {steps_per_second:.2f} steps/s, {samples_per_second:.0f} samples/s\n"
                    )
                    f.write(
                        f"  Epoch: {epoch + 1} ({epoch_progress_pct:.1f}%), Epoch ETC: {epoch_eta_str}\n"
                    )
                    if task_losses:
                        f.write(f"  Task Averages: {task_losses}\n")
                    f.write("\n")

        step += 1
        steps_in_current_epoch += 1

    # End of epoch
    epoch_steps = step - epoch_start_step
    print(f"\nCOMPLETED EPOCH {epoch + 1}")
    print(f"   Steps completed: {epoch_steps:,}")
    print(f"   Total steps so far: {step:,}")

    # Save checkpoint at end of epoch
    if accelerator.is_main_process:
        epoch_checkpoint_path = f"{output_dir}/epoch-{epoch + 1}"
        print(f"Saving epoch checkpoint to {epoch_checkpoint_path}...")
        model.save_checkpoint(f"{epoch_checkpoint_path}.pt")
        print(f"Epoch checkpoint saved")

    # Reset epoch counter and reinitialize all iterators for next epoch
    steps_in_current_epoch = 0
    epoch += 1

    if epoch < args.num_epochs:  # Don't reinitialize on the last epoch
        print(f"Preparing for next epoch...")
        # Reinitialize iterators for next epoch
        for task_name in iters.keys():
            dataloader_mapping = {
                "triplet": "triplet",
                "mlm": "mlm",
                "tone": "regression",
                "themes": "multilabel",
            }
            dl_name = dataloader_mapping.get(task_name, task_name)
            if dl_name and dataloaders.get(dl_name):
                iters[task_name] = iter(dataloaders[dl_name])
        print(f"All iterators reinitialized for epoch {epoch + 1}")

    print("=" * 50)

# Clean shutdown
print(f"\nTraining completed! Cleaning up...")
accelerator.end_training()
print("Accelerator shutdown complete")

# Save final model
if accelerator.is_main_process:
    final_model_path = f"{output_dir}/final_model"
    print(f"Saving final model to {final_model_path}...")
    model.save_checkpoint(f"{final_model_path}.pt")
    print(f"Final model saved")

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"FINAL STATISTICS:")
print(f"   - Total epochs completed: {args.num_epochs}")
print(f"   - Total steps executed: {step:,}")
print(f"   - Max steps per epoch: {max_steps_per_epoch:,}")
print(f"   - Tasks trained: {', '.join(available_tasks)}")
print(f"   - Final model saved to: {output_dir}/final_model")
print("=" * 80)
