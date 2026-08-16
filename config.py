from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import yaml
from gradient_diagnostics import GradientDiagnosticsConfig
import time
from datetime import timedelta


@dataclass
class TaskSpec:
    """Which rows of the corpus each task sees, and how its labels are built."""

    dataset_name: str
    split: str = "train"
    text_col: str = "textString"
    # MLM
    mlm_probability: float = 0.15
    themes_path: Optional[str] = None
    max_triplet_samples: int = 16
    # Subsampling: drop rows where V2Themes or V2Tone is missing/empty
    require_nonempty_themes_and_tone: bool = False
    # Directory for cached row-index artifacts (one-time build, mmap'd on reuse)
    index_cache_dir: str = "./cache"


@dataclass
class TrainArgs:
    # pretraining args
    model_name: str = "roberta-base"
    num_epochs: int = 3  # Number of epochs to train for
    warmup_ratio: float = 0.06
    batch_size: int = 32
    log_every: int = 5000
    gradient_accumulation_steps: int = 1
    # None runs `num_epochs` to completion.
    max_steps: Optional[int] = None
    # dataloader args
    dataloader_num_workers: int = 4
    pin_memory: bool = True 


@dataclass
class LossWeights:
    """
    Weights scale `total_loss` only; the per-task losses that get logged and
    diagnosed stay raw, so numbers remain comparable across weightings.

    The fields must mirror `model.DEFAULT_LOSS_WEIGHTS`: `setup_model` passes
    `as_dict()` straight into `MultiTaskRoberta`, which rejects unknown names.
    There is deliberately no `bias` weight -- that task exists only in
    fine-tuning, where it is the sole objective.
    """

    triplet: float = 1.0
    themes: float = 1.0
    tone: float = 1.0
    mlm: float = 1.0

    def as_dict(self) -> dict:
        return {
            "triplet": self.triplet,
            "themes": self.themes,
            "tone": self.tone,
            "mlm": self.mlm,
        }

    def is_default(self) -> bool:
        return all(w == 1.0 for w in self.as_dict().values())


@dataclass
class RunConfig:
    """Schema of a run_configs/*.yaml file — one file per pretraining run."""

    output_dir: str
    tasks: List[str] = field(default_factory=lambda: ["triplet", "mlm"])
    theme_count: int = 2000
    base_lr: float = 5e-5
    init_from_checkpoint: Optional[str] = None
    train_args: TrainArgs = field(default_factory=TrainArgs)
    task_spec: TaskSpec = field(
        default_factory=lambda: TaskSpec(
            dataset_name="upasanachatterjee/bignewsalign-with-gdelt",
            themes_path="top_themes.txt",
            max_triplet_samples=8,
        )
    )
    loss_weights: LossWeights = field(default_factory=LossWeights)
    # Multi-task gradient diagnostic; disabled by default so normal training
    # behaviour is unchanged.
    gradient_diagnostics: GradientDiagnosticsConfig = field(
        default_factory=GradientDiagnosticsConfig
    )


def load_run_config(path: str) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    train_args = TrainArgs(**raw.pop("train_args", {}))
    task_spec = TaskSpec(**raw.pop("task_spec", {}))
    grad_diag = GradientDiagnosticsConfig(**(raw.pop("gradient_diagnostics", {}) or {}))
    loss_weights = LossWeights(**(raw.pop("loss_weights", {}) or {}))
    return RunConfig(
        train_args=train_args,
        task_spec=task_spec,
        loss_weights=loss_weights,
        gradient_diagnostics=grad_diag,
        **raw,
    )


@dataclass
class TrainingProgress:
    """Wall-clock progress for one logging interval.

    Durations come pre-formatted for printing; `elapsed_seconds` is the raw
    value, for the TSV.
    """

    elapsed: str
    elapsed_seconds: float
    eta: str
    epoch_eta: str
    epoch_pct: float
    steps_per_second: float


def training_progress(
    training_start_time: float,
    epoch_start_time: float,
    step_times: Sequence[float],
    step: int,
    steps_in_current_epoch: int,
    max_steps_per_epoch: int,
    total_steps: int,
) -> TrainingProgress:
    """Estimate remaining time from recent step durations.

    The run-level ETA needs at least 5 recorded steps and the epoch ETA needs 1%
    of the epoch done; both report "..." until then.
    """
    now = time.time()
    elapsed_seconds = now - training_start_time
    elapsed = str(timedelta(seconds=int(elapsed_seconds)))

    if len(step_times) >= 5:
        avg_step_time = sum(step_times) / len(step_times)
        eta = str(timedelta(seconds=int((total_steps - step) * avg_step_time)))
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
    else:
        eta = "..."
        steps_per_second = 0.0

    epoch_elapsed = now - epoch_start_time
    epoch_progress = steps_in_current_epoch / max_steps_per_epoch
    if epoch_progress > 0.01:  # Avoid dividing by a near-zero fraction
        epoch_eta = str(
            timedelta(seconds=int((epoch_elapsed / epoch_progress) - epoch_elapsed))
        )
    else:
        epoch_eta = "..."

    return TrainingProgress(
        elapsed=elapsed,
        elapsed_seconds=elapsed_seconds,
        eta=eta,
        epoch_eta=epoch_eta,
        epoch_pct=epoch_progress * 100,
        steps_per_second=steps_per_second,
    )
