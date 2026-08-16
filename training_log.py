"""Per-run training logs.

Two files, both written into the run's ``output_dir``:

``training_log.txt``  human-readable -- the run summary, then one block per
                      logging interval (progress line + task losses).
``losses.tsv``        tidy per-interval losses, the file to plot from. One row
                      per logging interval, two columns per task::

                          import pandas as pd
                          df = pd.read_csv("losses.tsv", sep="\\t")
                          df.plot(x="step", y=["triplet", "mlm", "tone", "themes"])

The ``<task>`` columns are the RAW, unweighted per-task losses (see
``MultiTaskRoberta.forward``), so they stay comparable across loss weightings.
The ``<task>_weighted`` columns are those same losses times their
``loss_weights`` entry -- the contribution each objective actually made to the
total the optimizer stepped on. Under the default all-1.0 weighting the two are
identical; when they are not, the raw column says what the objective is doing
and the weighted column says how much of it the optimizer heard.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence

from config import TrainingProgress


class TrainingLogger:
    """Writes ``training_log.txt`` and ``losses.tsv`` for one run.

    Both files are truncated on construction. Diagnostic-only runs do no
    training, so they pass ``enabled=False`` to leave an earlier run's logs
    intact; every method then becomes a no-op.

    Only rank 0 should hold one of these.
    """

    def __init__(
        self,
        output_dir: str,
        tasks: Sequence[str],
        summary_lines: Iterable[str],
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.tasks = list(tasks)
        self.txt_path = os.path.join(output_dir, "training_log.txt")
        self.tsv_path = os.path.join(output_dir, "losses.tsv")
        if not self.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        with open(self.txt_path, "w") as handle:
            handle.write("=" * 78 + "\n")
            for line in summary_lines:
                handle.write(line + "\n")
            handle.write("=" * 78 + "\n\n")

        with open(self.tsv_path, "w") as handle:
            handle.write(
                "\t".join(
                    [
                        "step",
                        "epoch",
                        "lr",
                        "elapsed_s",
                        "steps_per_s",
                        *self.tasks,
                        *(f"{task}_weighted" for task in self.tasks),
                    ]
                )
                + "\n"
            )

    def log_interval(
        self,
        step: int,
        epoch: int,
        num_epochs: int,
        total_steps: int,
        lr: float,
        progress: TrainingProgress,
        task_losses: Dict[str, float],
        weighted_losses: Optional[Dict[str, float]] = None,
    ) -> None:
        """Print one interval to stdout and append it to both files."""
        weighted_losses = weighted_losses or {}
        header = (
            f"step {step:,}/{total_steps:,} | epoch {epoch}/{num_epochs} "
            f"{progress.epoch_pct:5.1f}% | lr {lr:.2e} | "
            f"elapsed {progress.elapsed} | eta {progress.eta} "
            f"(epoch {progress.epoch_eta}) | {progress.steps_per_second:.2f} steps/s"
        )
        losses = "  loss  " + "  ".join(
            f"{task}={value:.4f}" for task, value in task_losses.items()
        )
        block = [header, losses]
        # Only worth a second line when a weight is actually doing something;
        # under the default all-1.0 weighting it would repeat the line above.
        if any(
            abs(weighted_losses.get(task, value) - value) > 1e-9
            for task, value in task_losses.items()
        ):
            block.append(
                "  wtd   "
                + "  ".join(
                    f"{task}={weighted_losses[task]:.4f}"
                    for task in task_losses
                    if task in weighted_losses
                )
            )
        for line in block:
            print(line, flush=True)

        if not self.enabled:
            return

        with open(self.txt_path, "a") as handle:
            for line in block:
                handle.write(line + "\n")

        row: List[str] = [
            str(step),
            str(epoch),
            f"{lr:.6e}",
            f"{progress.elapsed_seconds:.1f}",
            f"{progress.steps_per_second:.3f}",
        ]
        # A task with no loss this interval leaves its cell empty rather than 0,
        # so pandas reads it as NaN and the curve breaks instead of dipping.
        for source in (task_losses, weighted_losses):
            for task in self.tasks:
                value: Optional[float] = source.get(task)
                row.append("" if value is None else f"{value:.6f}")
        with open(self.tsv_path, "a") as handle:
            handle.write("\t".join(row) + "\n")

    def note(self, message: str) -> None:
        """Append a one-off line (epoch boundary, checkpoint) to the text log."""
        print(message, flush=True)
        if self.enabled:
            with open(self.txt_path, "a") as handle:
                handle.write(message + "\n")
