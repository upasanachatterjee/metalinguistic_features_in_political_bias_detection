from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
from datasets import load_dataset, Dataset
import os
from huggingface_hub import login
import time
from datetime import datetime, timedelta


@dataclass
class TaskSpec:
    dataset_name: str
    split: str = "train"
    text_col: str = "textString"
    # MLM
    mlm_probability: float = 0.15
    # Multilabel
    multi_label_col: Optional[str] = (
        "V2Themes"  # column containing list or string labels
    )
    themes_path: Optional[str] = None
    # Regression
    regression_col: Optional[str] = "V2Tone"  # column containing float target(s)
    tones_count: int = 2


@dataclass
class TrainArgs:
    # pretraining args
    model_name: str = "roberta-base"
    num_epochs: int = 2  # Number of epochs to train for
    warmup_ratio: float = 0.1
    batch_size: int = 8 
    log_every: int = 5000
    # dataloader args
    dataloader_num_workers: int = 4  # More workers for high-throughput GPUs
    pin_memory: bool = True  # Faster CPU->GPU transfer


def login_to_huggingface(token):
    login(token)
    print("Logged in to Hugging Face Hub")


def update_loss(
    loss_item: Optional[torch.Tensor], global_loss: Optional[torch.Tensor]
) -> torch.Tensor | None:
    """Helper to update global loss with a new loss item."""
    if loss_item is not None:
        if global_loss is None:
            return loss_item
        else:
            return global_loss + loss_item
    return global_loss


def calculate_eta(
    training_start_time: float,
    epoch_start_time: float,
    step_times: List[float],
    step: int,
    steps_in_current_epoch: int,
    max_steps_per_epoch: int,
    TOTAL_STEPS: int,
    avg_step_time: float,
    batch_size: int,
    num_processes: int,
) -> Tuple[str, str, str, str, float, float, float, float, float]:
    """Calculate estimated time of arrival (completion) for training."""
    # Calculate timing statistics
    current_time = time.time()
    elapsed_time = current_time - training_start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))

    # Calculate ETC based on recent step times
    if len(step_times) >= 5:  # Need at least 5 steps for reliable estimate
        avg_step_time = sum(step_times) / len(step_times)
        remaining_steps = TOTAL_STEPS - step
        eta_seconds = remaining_steps * avg_step_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        completion_time = datetime.now() + timedelta(seconds=eta_seconds)
        completion_str = completion_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        eta_str = "Calculating..."
        completion_str = "Calculating..."

    # Calculate epoch ETC
    epoch_elapsed = current_time - epoch_start_time
    epoch_progress = steps_in_current_epoch / max_steps_per_epoch
    if epoch_progress > 0.01:  # Avoid division by very small numbers
        epoch_eta_seconds = (epoch_elapsed / epoch_progress) - epoch_elapsed
        epoch_eta_str = str(timedelta(seconds=int(epoch_eta_seconds)))
    else:
        epoch_eta_str = "Calculating..."

    # Calculate steps per second
    if len(step_times) >= 5 and avg_step_time > 0:
        steps_per_second = 1.0 / avg_step_time
        samples_per_second = steps_per_second * batch_size * num_processes
    else:
        steps_per_second = 0
        samples_per_second = 0

    progress_pct = (step / TOTAL_STEPS) * 100
    epoch_progress = steps_in_current_epoch / max_steps_per_epoch
    epoch_progress_pct = epoch_progress * 100

    return (
        eta_str,
        completion_str,
        elapsed_str,
        epoch_eta_str,
        progress_pct,
        epoch_progress_pct,
        steps_per_second,
        samples_per_second,
        avg_step_time,
    )
