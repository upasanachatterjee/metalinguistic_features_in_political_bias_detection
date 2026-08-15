"""Temporary diagnostic utility for multi-task gradient analysis.

Measures, for each task loss returned by ``MultiTaskRoberta``:

1. the magnitude of the gradient it produces on the *shared* transformer
   representation (by default the final RoBERTa encoder layer);
2. the pairwise alignment / conflict between those task gradients.

Outputs (all written to the run's ``output_dir``)
-------------------------------------------------
``gradient_diagnostics.txt``   human-readable report (tables + interpretation)
``gradient_diagnostics.tsv``   tidy long-format measurements, one row per
                               (virtual batch, metric); the file for plots and
                               paper tables. Concatenates cleanly across runs
                               because every row carries the knob settings.
``gradient_diagnostics.json``  the full run record: config knobs, environment,
                               every per-virtual-batch measurement, summary
                               statistics and suggested weights -- everything a
                               reader needs to reproduce the numbers.

Reading the TSV::

    import pandas as pd, glob
    df = pd.concat(
        pd.read_csv(p, sep="\\t", comment="#") for p in glob.glob("*/gradient_diagnostics.tsv")
    )
    norms = df[df.metric == "grad_norm"]
    norms.pivot_table(index="task_a", columns="parameter_scope", values="value",
                      aggfunc="median")
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Numerical floor for cosine denominators.
EPS = 1e-12

# Logical task name -> exact key returned by MultiTaskRoberta.forward().
# Verified against model.py: the model emits "triplet_loss", "theme_loss",
# "tone_loss", "bias_loss" (only when num_bias_classes is set) and "mlm_loss",
# plus the summed "loss".
TASK_LOSS_KEYS: Dict[str, str] = {
    "triplet": "triplet_loss",
    "themes": "theme_loss",
    "tone": "tone_loss",
    "bias": "bias_loss",
    "mlm": "mlm_loss",
}

# Logical task name -> batch key used to count examples per microbatch, so the
# printout can confirm each virtual batch really covers ~32 examples.
TASK_BATCH_KEYS: Dict[str, str] = {
    "triplet": "triplet_a_ids",  # anchors; each anchor also pulls a pos + neg
    "themes": "theme_input_ids",
    "tone": "tone_input_ids",
    "bias": "bias_input_ids",
    "mlm": "mlm_input_ids",
}

# Columns of the tidy TSV. Every row carries the knob settings so that TSVs
# from separate runs (different scope / batch size / K) can be concatenated and
# grouped without a separate join.
TSV_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "run_label",
    "parameter_scope",
    "num_diag_params",
    "train_batch_size",
    "microbatches_per_virtual_batch",
    "examples_per_virtual_batch",
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
#   diag_secs   seconds spent in torch.autograd.grad for the virtual batch
#   wall_secs   wall-clock seconds for the virtual batch (fwd + diagnostic)
#   peak_mem_mib  peak CUDA memory during the virtual batch, MiB (CUDA only)
# Gradient norms are ||(1/K) sum_k g_k||, NOT (1/K) sum_k ||g_k||.
# Load with: pd.read_csv(path, sep="\\t", comment="#")
"""


@dataclass
class GradientDiagnosticsConfig:
    """YAML-configurable knobs for the gradient diagnostic.

    ``microbatches_per_virtual_batch`` should be 32 // train_args.batch_size so
    each virtual batch approximates the normal 32-example global batch.

    ``run_label`` names the run in the TSV/JSON so a knob sweep (scope, batch
    size, K, ...) can be grouped and plotted. ``records_path`` optionally
    appends every row to a shared sweep file as well as the per-run TSV.
    """

    enabled: bool = False
    microbatches_per_virtual_batch: int = 8
    num_virtual_batches: int = 5
    parameter_scope: str = "final_encoder_layer"
    diagnostic_only: bool = True
    # Global batch size the diagnostic is trying to emulate; only used to
    # sanity-check microbatches_per_virtual_batch against train_args.batch_size.
    target_global_batch_size: int = 32
    run_label: Optional[str] = None
    records_path: Optional[str] = None


def get_diagnostic_parameters(
    base_model: torch.nn.Module, scope: str
) -> Tuple[List[torch.nn.Parameter], List[str]]:
    """Return the shared-representation parameters to differentiate against.

    ``base_model`` must be the unwrapped model (``accelerator.unwrap_model``).
    Scopes are intentionally pluggable so the analysis can be widened later:

    - ``final_encoder_layer``   -- backbone.encoder.layer[-1] (~7M params)
    - ``last_N_encoder_layers`` -- e.g. ``last_4_encoder_layers``
    - ``full_backbone``         -- the whole RoBERTa backbone (~110M params;
                                   ~440MB of CPU float32 per task accumulator)
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
    if not params:
        raise ValueError(f"No trainable parameters found for scope '{scope}'.")
    return params, names


def count_batch_examples(combined_batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Per-task example counts for one microbatch (for verification output)."""
    counts: Dict[str, int] = {}
    for task, key in TASK_BATCH_KEYS.items():
        tensor = combined_batch.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) > 0:
            counts[task] = int(tensor.shape[0])
    return counts


def round_weight(weight: float, sig_figs: int = 2) -> float:
    """Round a suggested weight to `sig_figs` significant figures.

    Suggested weights are only meant to correct order-of-magnitude imbalances,
    so the paper (and the run YAML) should quote rounded values.
    """
    if weight == 0 or not math.isfinite(weight):
        return 0.0
    digits = -int(math.floor(math.log10(abs(weight)))) + (sig_figs - 1)
    return round(weight, max(digits, 0))


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
    ) -> None:
        if accelerator.num_processes != 1:
            raise RuntimeError(
                "Gradient diagnostics must be run with a single Accelerate process."
            )

        self.accelerator = accelerator
        self.config = config
        self.output_dir = output_dir
        self.train_batch_size = train_batch_size
        self.metadata = dict(metadata or {})

        base_model = accelerator.unwrap_model(model)
        self.parameters, param_names = get_diagnostic_parameters(
            base_model, config.parameter_scope
        )
        self.num_params = sum(p.numel() for p in self.parameters)
        self.applied_loss_weights = dict(getattr(base_model, "loss_weights", {}) or {})

        self.started_at = datetime.now()
        self.run_id = self.started_at.strftime("%Y%m%dT%H%M%S")
        self.run_label = config.run_label or (
            f"{config.parameter_scope}_bs{train_batch_size}_"
            f"k{config.microbatches_per_virtual_batch}"
        )
        self.examples_per_virtual_batch = (
            train_batch_size * config.microbatches_per_virtual_batch
            if train_batch_size is not None
            else None
        )

        # Per-virtual-batch accumulators (all CPU float32, fully detached).
        self._grad_sums: Dict[str, torch.Tensor] = {}
        self._task_microbatches: Dict[str, int] = {}
        self._loss_sums: Dict[str, float] = {}
        self._example_counts: Dict[str, int] = {}
        self._microbatches = 0
        self._diag_seconds = 0.0
        self._vb_start = time.perf_counter()

        # Collected results across virtual batches.
        self.norm_history: Dict[str, List[float]] = {}
        self.cosine_history: Dict[Tuple[str, str], List[float]] = {}
        self.loss_history: Dict[str, List[float]] = {}
        self.virtual_batches: List[Dict[str, Any]] = []
        self.completed_virtual_batches = 0

        self._warned: set = set()
        self._lines: List[str] = []
        self._rows: List[Tuple[Any, ...]] = []

        self._emit("=" * 78)
        self._emit("GRADIENT DIAGNOSTICS (temporary; training objective unchanged)")
        self._emit("=" * 78)
        self._emit(f"  run id                       : {self.run_id}")
        self._emit(f"  run label                    : {self.run_label}")
        self._emit(f"  parameter scope              : {config.parameter_scope}")
        self._emit(
            f"  diagnosed parameters         : {len(self.parameters)} tensors, "
            f"{self.num_params:,} scalars"
        )
        self._emit(
            f"    first/last tensor          : {param_names[0]} ... {param_names[-1]}"
        )
        self._emit(
            f"  microbatches / virtual batch : {config.microbatches_per_virtual_batch}"
        )
        self._emit(f"  virtual batches to collect   : {config.num_virtual_batches}")
        self._emit(f"  diagnostic_only              : {config.diagnostic_only}")
        self._emit(
            f"  loss weights in effect       : {self.applied_loss_weights or 'n/a'}"
        )
        for key, value in self.metadata.items():
            self._emit(f"  {key:<29}: {value}")
        if train_batch_size is not None:
            self._emit(
                f"  implied examples / virtual batch: {train_batch_size} x "
                f"{config.microbatches_per_virtual_batch} = "
                f"{self.examples_per_virtual_batch}"
            )
            if self.examples_per_virtual_batch != config.target_global_batch_size:
                self._emit(
                    f"  WARNING: that is not the target global batch size "
                    f"({config.target_global_batch_size}). Set "
                    f"microbatches_per_virtual_batch = "
                    f"{config.target_global_batch_size} // batch_size."
                )
        if getattr(accelerator, "mixed_precision", "no") == "fp16":
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

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------
    def is_complete(self) -> bool:
        return self.completed_virtual_batches >= self.config.num_virtual_batches

    def record_microbatch(
        self,
        outputs: Dict[str, torch.Tensor],
        combined_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Compute per-task gradients for one microbatch.

        Must be called *after* the forward pass and *before*
        ``accelerator.backward(total_loss)``. Uses ``retain_graph=True`` so the
        normal combined backward can still run afterwards.
        """
        if self.is_complete():
            return

        started = time.perf_counter()

        if combined_batch is not None:
            for task, count in count_batch_examples(combined_batch).items():
                self._example_counts[task] = self._example_counts.get(task, 0) + count

        for task, loss_key in TASK_LOSS_KEYS.items():
            loss = outputs.get(loss_key)
            if loss is None:
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

            if task in self._grad_sums:
                self._grad_sums[task] += vector
            else:
                self._grad_sums[task] = vector
            self._task_microbatches[task] = self._task_microbatches.get(task, 0) + 1
            self._loss_sums[task] = self._loss_sums.get(task, 0.0) + float(loss.item())

        self._diag_seconds += time.perf_counter() - started
        self._microbatches += 1
        if self._microbatches >= self.config.microbatches_per_virtual_batch:
            self._finalize_virtual_batch()

    def _flatten(
        self, task: str, grads: Iterable[Optional[torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        """Detach, flatten and move one task's gradients to a CPU float32 vector."""
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
        self.completed_virtual_batches += 1
        index = self.completed_virtual_batches
        wall_seconds = time.perf_counter() - self._vb_start

        mean_grads: Dict[str, torch.Tensor] = {}
        for task, grad_sum in self._grad_sums.items():
            count = self._task_microbatches[task]
            if count != k:
                self._emit(
                    f"  NOTE: {task} contributed {count}/{k} microbatches in "
                    f"virtual batch {index}; averaging over {count}."
                )
            # ||mean gradient||, NOT mean of per-microbatch norms.
            mean_grads[task] = grad_sum / float(count)

        record: Dict[str, Any] = {
            "virtual_batch": index,
            "grad_norm": {},
            "cosine": {},
            "loss": {},
            "examples": dict(self._example_counts),
            "microbatches": self._microbatches,
            "diag_seconds": round(self._diag_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }

        for task, mean_grad in mean_grads.items():
            norm = float(torch.linalg.vector_norm(mean_grad).item())
            self.norm_history.setdefault(task, []).append(norm)
            record["grad_norm"][task] = norm
            self._row(index, "grad_norm", task, "", norm)

            mean_loss = self._loss_sums[task] / self._task_microbatches[task]
            self.loss_history.setdefault(task, []).append(mean_loss)
            record["loss"][task] = mean_loss
            self._row(index, "loss", task, "", mean_loss)

        for task_a, task_b in combinations(sorted(mean_grads), 2):
            g_a, g_b = mean_grads[task_a], mean_grads[task_b]
            denom = (
                torch.linalg.vector_norm(g_a) * torch.linalg.vector_norm(g_b)
            ).clamp_min(EPS)
            cosine = float((torch.dot(g_a, g_b) / denom).item())
            self.cosine_history.setdefault((task_a, task_b), []).append(cosine)
            record["cosine"][f"{task_a}|{task_b}"] = cosine
            self._row(index, "cosine", task_a, task_b, cosine)

        for task, count in self._example_counts.items():
            self._row(index, "examples", task, "", count)
        self._row(index, "diag_secs", "", "", round(self._diag_seconds, 4))
        self._row(index, "wall_secs", "", "", round(wall_seconds, 4))
        if torch.cuda.is_available():
            peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
            record["peak_mem_mib"] = round(peak_mib, 1)
            self._row(index, "peak_mem_mib", "", "", round(peak_mib, 1))
            torch.cuda.reset_peak_memory_stats()

        self.virtual_batches.append(record)

        examples = " ".join(
            f"{task}={count}" for task, count in sorted(self._example_counts.items())
        )
        norms = " ".join(
            f"{task}={self.norm_history[task][-1]:.4f}" for task in sorted(mean_grads)
        )
        self._emit(
            f"[virtual batch {index}/{self.config.num_virtual_batches}] "
            f"microbatches={self._microbatches} examples: {examples} "
            f"({wall_seconds:.1f}s wall, {self._diag_seconds:.1f}s in autograd.grad)"
        )
        self._emit(f"    ||mean grad||: {norms}")

        # Reset accumulators for the next virtual batch.
        self._grad_sums = {}
        self._task_microbatches = {}
        self._loss_sums = {}
        self._example_counts = {}
        self._microbatches = 0
        self._diag_seconds = 0.0
        self._vb_start = time.perf_counter()

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Summary statistics per task and per task pair (also used for JSON)."""
        norms = {
            task: {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
            for task, values in self.norm_history.items()
        }
        cosines = {
            f"{a}|{b}": {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
            for (a, b), values in self.cosine_history.items()
        }
        losses = {
            task: {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
            }
            for task, values in self.loss_history.items()
        }
        return {"grad_norm": norms, "cosine": cosines, "loss": losses}

    def suggested_weights(self) -> Dict[str, Any]:
        """target / median_norm[task], raw and rounded to 2 significant figures."""
        medians = {task: statistics.median(v) for task, v in self.norm_history.items()}
        if not medians:
            return {}
        target = statistics.median(list(medians.values()))
        raw = {task: target / max(m, EPS) for task, m in medians.items()}
        return {
            "target": target,
            "raw": raw,
            "rounded": {task: round_weight(w) for task, w in raw.items()},
        }

    def report(self) -> None:
        self._emit("")
        self._emit("=" * 78)
        self._emit(
            f"SUMMARY over {self.completed_virtual_batches} virtual batches "
            f"(scope: {self.config.parameter_scope})"
        )
        self._emit("=" * 78)

        if not self.norm_history:
            self._emit("No task gradients were collected.")
            self._write_outputs()
            return

        stats = self.summary()

        self._emit("")
        self._emit("Gradient norm summary")
        self._emit("")
        self._emit(f"{'task':<20}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}")
        self._emit("-" * 60)
        for task in sorted(stats["grad_norm"]):
            row = stats["grad_norm"][task]
            self._emit(
                f"{task:<20}{row['median']:>10.4f}{row['mean']:>10.4f}"
                f"{row['min']:>10.4f}{row['max']:>10.4f}"
            )
        self._emit("")
        self._emit(
            "Median is the primary comparison (few virtual batches were sampled)."
        )

        if stats["cosine"]:
            self._emit("")
            self._emit("Gradient cosine summary")
            self._emit("")
            self._emit(f"{'task A':<20}{'task B':<20}{'median':>10}{'mean':>10}")
            self._emit("-" * 60)
            for pair in sorted(stats["cosine"]):
                task_a, task_b = pair.split("|")
                row = stats["cosine"][pair]
                self._emit(
                    f"{task_a:<20}{task_b:<20}"
                    f"{row['median']:>+10.3f}{row['mean']:>+10.3f}"
                )
            self._emit("")
            self._emit("Interpretation (conservative):")
            self._emit("  near 0            : largely unrelated gradients")
            self._emit("  positive          : objectives push in similar directions")
            self._emit("  consistently neg. : potential conflict")
            self._emit("  repeatedly <= -0.3: worth investigating")
            self._emit(
                "  A single negative measurement is NOT evidence of task conflict."
            )

        weights = self.suggested_weights()
        self._emit("")
        self._emit("Suggested relative weights (diagnostic suggestion only)")
        self._emit("")
        self._emit(f"{'task':<20}{'raw':>10}{'rounded':>10}")
        self._emit("-" * 40)
        for task in sorted(weights["raw"], key=lambda t: -weights["raw"][t]):
            self._emit(
                f"{task:<20}{weights['raw'][task]:>10.3f}"
                f"{weights['rounded'][task]:>10.2f}"
            )
        self._emit("")
        self._emit(
            f"  target = median of task median norms = {weights['target']:.4f}; "
            "raw_weight[task] = target / median_norm[task]."
        )
        self._emit(
            "  Use the rounded column, and only to correct large magnitude "
            "imbalances. These are NOT optimal hyperparameters, and no loss "
            "reweighting has been applied to the training objective."
        )
        self._emit("")
        self._emit("  Copy into the run YAML to test them (placeholders are all 1.0):")
        self._emit("")
        self._emit("    loss_weights:")
        for task in ("triplet", "themes", "tone", "bias", "mlm"):
            value = weights["rounded"].get(task)
            suffix = "" if value is not None else "   # task not measured"
            self._emit(f"      {task}: {value if value is not None else 1.0}{suffix}")
        self._emit("=" * 78)
        self._write_outputs()

    # ------------------------------------------------------------------
    # output files
    # ------------------------------------------------------------------
    def _row(
        self, virtual_batch: int, metric: str, task_a: str, task_b: str, value: float
    ) -> None:
        self._rows.append(
            (
                self.run_id,
                self.run_label,
                self.config.parameter_scope,
                self.num_params,
                self.train_batch_size if self.train_batch_size is not None else "",
                self.config.microbatches_per_virtual_batch,
                self.examples_per_virtual_batch
                if self.examples_per_virtual_batch is not None
                else "",
                virtual_batch,
                metric,
                task_a,
                task_b,
                value,
            )
        )

    def _json_record(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "config": asdict(self.config),
            "setup": {
                "train_batch_size": self.train_batch_size,
                "examples_per_virtual_batch": self.examples_per_virtual_batch,
                "num_diag_params": self.num_params,
                "num_diag_tensors": len(self.parameters),
                "loss_weights_in_effect": self.applied_loss_weights,
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
                "argv": sys.argv,
            },
            "method": {
                "virtual_batch_gradient": "mean of per-microbatch gradient vectors",
                "norm": "L2 norm of the mean gradient vector (not mean of norms)",
                "cosine_epsilon": EPS,
                "gradients_via": "torch.autograd.grad(retain_graph=True, allow_unused=True)",
                "losses_reported": "raw, unweighted per-task losses",
            },
            "virtual_batches": self.virtual_batches,
            "summary": self.summary(),
            "suggested_weights": self.suggested_weights(),
        }

    def _write_outputs(self) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        txt_path = os.path.join(self.output_dir, "gradient_diagnostics.txt")
        with open(txt_path, "w") as handle:
            handle.write("\n".join(self._lines) + "\n")

        tsv_path = os.path.join(self.output_dir, "gradient_diagnostics.tsv")
        with open(tsv_path, "w") as handle:
            handle.write(TSV_HEADER_COMMENT)
            handle.write("\t".join(TSV_COLUMNS) + "\n")
            for row in self._rows:
                handle.write("\t".join(str(v) for v in row) + "\n")

        json_path = os.path.join(self.output_dir, "gradient_diagnostics.json")
        with open(json_path, "w") as handle:
            json.dump(self._json_record(), handle, indent=2)
            handle.write("\n")

        written = [txt_path, tsv_path, json_path]

        # Optional shared sweep file: append so several runs (different scopes,
        # batch sizes, K) land in one table for cross-run plots.
        if self.config.records_path:
            shared = self.config.records_path
            os.makedirs(os.path.dirname(shared) or ".", exist_ok=True)
            exists = os.path.exists(shared)
            with open(shared, "a") as handle:
                if not exists:
                    handle.write(TSV_HEADER_COMMENT)
                    handle.write("\t".join(TSV_COLUMNS) + "\n")
                for row in self._rows:
                    handle.write("\t".join(str(v) for v in row) + "\n")
            written.append(f"{shared} (appended)")

        print("\nGradient diagnostics written to:")
        for path in written:
            print(f"   {path}")

    # ------------------------------------------------------------------
    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._emit(message)

    def _emit(self, line: str) -> None:
        print(line)
        self._lines.append(line)
