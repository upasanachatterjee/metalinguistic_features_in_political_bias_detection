from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM

DEFAULT_LOSS_WEIGHTS = {
    "triplet": 1.0,
    "themes": 1.0,
    "tone": 1.0,
    "mlm": 1.0,
}


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_labels, hidden_size=768, classifier_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        # assume input is <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiTaskRoberta(nn.Module):
    """Shared RoBERTa backbone with one head per task.

    `forward` dispatches on which prefixed keys are present in kwargs
    (`triplet_*`, `theme_*`, `tone_*`, `mlm_*`), so a single call can
    carry any subset of the tasks; `pretraining.TASK_BATCH_KEYS` is the mapping
    from collator output to those keys. `bias_*` dispatches the same way but
    belongs to fine-tuning alone.

    `outputs["<task>_loss"]` is always the RAW, unweighted task loss;
    `outputs["<task>_weighted_loss"]` is that loss times its `loss_weights`
    entry, i.e. exactly the term that went into `outputs["loss"]`. Logging and
    the gradient diagnostic read the raw key, so the numbers stay comparable
    across weightings.
    """

    TRIPLET_MARGIN = 1.0

    def __init__(
        self,
        name="roberta-base",
        num_tones=1,
        num_themes=2000,
        num_bias_classes=None,
        loss_weights=None,
    ):
        super().__init__()
        self.name = name
        # Per-task loss weights
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        for task, weight in (loss_weights or {}).items():
            if task not in DEFAULT_LOSS_WEIGHTS:
                raise ValueError(
                    f"Unknown loss weight '{task}'. "
                    f"Expected one of {sorted(DEFAULT_LOSS_WEIGHTS)}."
                )
            self.loss_weights[task] = float(weight)

        # Single backbone model
        self.backbone = AutoModel.from_pretrained(name)
        hid = self.backbone.config.hidden_size

        # Task-specific heads
        self.theme_head = nn.Linear(hid, num_themes)
        self.tone_head = nn.Linear(hid, num_tones)

        # Add a new head for single-class classification
        self.num_bias_classes = num_bias_classes
        if self.num_bias_classes is not None:
            self.bias_head = ClassificationHead(num_bias_classes, hidden_size=hid)

        # MLM head - create just the head, not the full model
        # Load the full MLM model temporarily to get the LM head
        mlm_model = AutoModelForMaskedLM.from_pretrained(name)
        self.lm_head = mlm_model.lm_head
        # Clean up the temporary model
        del mlm_model

        # When True, `forward` also returns detached triplet geometry under
        # `outputs["triplet_stats"]`. Off during normal training: each entry is
        # a GPU->CPU sync that the training loop has no use for.
        self.collect_triplet_stats = False

        # Per-task objectives. Stateless, so they carry no state_dict entries.
        self.triplet_loss_fct = nn.TripletMarginLoss(margin=self.TRIPLET_MARGIN, p=2)
        self.theme_loss_fct = nn.BCEWithLogitsLoss()
        self.tone_loss_fct = nn.MSELoss()
        self.bias_loss_fct = nn.CrossEntropyLoss()
        self.mlm_loss_fct = nn.CrossEntropyLoss()

    # Add these methods to support gradient checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the backbone model"""
        # Reentrant checkpointing is incompatible with DDP when a module is
        # forwarded multiple times per step (we run the backbone for triplet
        # a/p/n + themes + tone + mlm), so force the non-reentrant variant.
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            # Fallback for older transformers versions
            self.backbone.config.use_cache = False
            self.backbone.gradient_checkpointing = True
        return self

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the backbone model"""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        else:
            # Fallback for older transformers versions
            self.backbone.config.use_cache = True
            self.backbone.gradient_checkpointing = False
        return self

    def forward(self, **kwargs):
        outputs = {}
        total_loss = torch.tensor(0.0, device=self.backbone.device)

        # --- Triplet Task ---
        if "triplet_a_ids" in kwargs and kwargs["triplet_a_ids"].shape[0] > 0:
            za = self.forward_single(
                kwargs["triplet_a_ids"], kwargs["triplet_a_mask"]
            )
            zp = self.forward_single(
                kwargs["triplet_p_ids"], kwargs["triplet_p_mask"]
            )
            zn = self.forward_single(
                kwargs["triplet_n_ids"], kwargs["triplet_n_mask"]
            )

            triplet_loss = self.triplet_loss_fct(za, zp, zn)
            weighted = self.loss_weights["triplet"] * triplet_loss
            total_loss += weighted
            outputs["triplet_loss"] = triplet_loss
            outputs["triplet_weighted_loss"] = weighted
            if self.collect_triplet_stats:
                outputs["triplet_stats"] = self._triplet_stats(za, zp, zn)

        # --- Classification Tasks (Themes & Tone) ---
        # Theme Task
        if "theme_labels" in kwargs and "theme_input_ids" in kwargs:
            input_ids, attention_mask = (
                kwargs["theme_input_ids"],
                kwargs["theme_attention_mask"],
            )
            pooled = self.forward_single(input_ids, attention_mask)
            theme_logits = self.theme_head(pooled)
            theme_loss = self.theme_loss_fct(
                theme_logits, kwargs["theme_labels"].float()
            )
            weighted = self.loss_weights["themes"] * theme_loss
            total_loss += weighted
            outputs["theme_loss"] = theme_loss
            outputs["theme_weighted_loss"] = weighted
            outputs["theme_logits"] = theme_logits

        # Tone Task
        if "tone_labels" in kwargs and "tone_input_ids" in kwargs:
            input_ids, attention_mask = (
                kwargs["tone_input_ids"],
                kwargs["tone_attention_mask"],
            )
            pooled = self.forward_single(input_ids, attention_mask)
            tone_logits = self.tone_head(pooled)
            tone_loss = self.tone_loss_fct(tone_logits, kwargs["tone_labels"].float())
            weighted = self.loss_weights["tone"] * tone_loss
            total_loss += weighted
            outputs["tone_loss"] = tone_loss
            outputs["tone_weighted_loss"] = weighted
            outputs["tone_logits"] = tone_logits

        # --- Single-Class Classification Task (e.g., Bias/Ideology) ---
        # Fine-tuning only, and there it is the ONLY objective
        if "bias_labels" in kwargs and "bias_input_ids" in kwargs:
            assert self.num_bias_classes is not None, (
                "num_bias_classes must be set during model initialization for classification."
            )
            input_ids, attention_mask = (
                kwargs["bias_input_ids"],
                kwargs["bias_attention_mask"],
            )
            pooled = self.forward_single(input_ids, attention_mask)
            bias_logits = self.bias_head(pooled)
            bias_loss = self.bias_loss_fct(bias_logits, kwargs["bias_labels"])
            total_loss += bias_loss
            outputs["bias_loss"] = bias_loss
            outputs["bias_logits"] = bias_logits

        # --- MLM Task ---
        if "mlm_input_ids" in kwargs:
            input_ids, attention_mask = (
                kwargs["mlm_input_ids"],
                kwargs["mlm_attention_mask"],
            )
            labels = kwargs["mlm_labels"]

            backbone_output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            sequence_output = backbone_output.last_hidden_state
            prediction_scores = self.lm_head(sequence_output)

            mlm_loss = self.mlm_loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1)
            )
            weighted = self.loss_weights["mlm"] * mlm_loss
            total_loss += weighted
            outputs["mlm_loss"] = mlm_loss
            outputs["mlm_weighted_loss"] = weighted
            outputs["mlm_logits"] = prediction_scores

        outputs["loss"] = total_loss
        return outputs

    def _triplet_stats(
        self, za: torch.Tensor, zp: torch.Tensor, zn: torch.Tensor
    ) -> Dict[str, float]:
        """Geometry behind the triplet loss, for the gradient diagnostic.

        `nn.TripletMarginLoss` reports only `relu(violation).mean()`, which hides
        why the triplet gradient is noisy: a batch of easy triplets (all
        violations negative) contributes nothing at all, and one with a handful
        of active triplets contributes a gradient built from those few. These
        numbers separate the two cases.
        """
        with torch.no_grad():
            d_pos = F.pairwise_distance(za, zp, p=2)
            d_neg = F.pairwise_distance(za, zn, p=2)
            violation = d_pos - d_neg + self.TRIPLET_MARGIN
            active = violation > 0
            return {
                "num_triplets": int(za.shape[0]),
                "active_fraction": float(active.float().mean().item()),
                "mean_positive_distance": float(d_pos.mean().item()),
                "mean_negative_distance": float(d_neg.mean().item()),
                "mean_violation": float(violation.mean().item()),
            }

    def forward_single(self, input_ids, attention_mask):
        """Helper for a single forward pass for non-MLM tasks."""
        backbone_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        # Use CLS token embedding
        cls_embedding = backbone_output.last_hidden_state[:, 0, :]
        return cls_embedding

    def save_checkpoint(self, path):
        """Save state dict plus the head sizes needed to rebuild the model."""
        config = {
            "name": self.name,
            "hidden_size": self.backbone.config.hidden_size,
            "vocab_size": self.backbone.config.vocab_size,
            "num_themes": self.theme_head.out_features,
            "num_tones": self.tone_head.out_features,
            "loss_weights": dict(self.loss_weights),
        }
        if self.num_bias_classes is not None:
            config["num_bias_classes"] = self.num_bias_classes

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, path)
