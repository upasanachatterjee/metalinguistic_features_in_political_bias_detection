import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM
import torch


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
    def __init__(
        self, name="roberta-base", num_tones=2, num_themes=2000, num_bias_classes=None
    ):
        super().__init__()
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

    # Add these methods to support gradient checkpointing
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the backbone model"""
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
        # print(f"going forward w kwargs: {kwargs.keys()}")
        outputs = {}
        total_loss = torch.tensor(0.0, device=self.backbone.device)

        # --- Triplet Task ---
        if "a_ids" in kwargs and "p_ids" in kwargs and "n_ids" in kwargs:
            a_ids, a_mask = kwargs["a_ids"], kwargs["a_mask"]
            p_ids, p_mask = kwargs["p_ids"], kwargs["p_mask"]
            n_ids, n_mask = kwargs["n_ids"], kwargs["n_mask"]

            za = self.forward_single(a_ids, a_mask)
            zp = self.forward_single(p_ids, p_mask)
            zn = self.forward_single(n_ids, n_mask)

            triplet_loss_fct = nn.TripletMarginLoss(margin=1.0, p=2)
            triplet_loss = triplet_loss_fct(za, zp, zn)
            total_loss += triplet_loss
            outputs["triplet_loss"] = triplet_loss

        # --- Classification Tasks (Themes & Tone) ---
        # Theme Task
        if "theme_labels" in kwargs and "theme_input_ids" in kwargs:
            input_ids, attention_mask = (
                kwargs["theme_input_ids"],
                kwargs["theme_attention_mask"],
            )
            pooled = self.forward_single(input_ids, attention_mask)
            theme_logits = self.theme_head(pooled)
            theme_loss_fct = nn.BCEWithLogitsLoss()
            theme_loss = theme_loss_fct(theme_logits, kwargs["theme_labels"].float())
            total_loss += theme_loss
            outputs["theme_loss"] = theme_loss
            outputs["theme_logits"] = theme_logits

        # Tone Task
        if "tone_labels" in kwargs and "tone_input_ids" in kwargs:
            input_ids, attention_mask = (
                kwargs["tone_input_ids"],
                kwargs["tone_attention_mask"],
            )
            pooled = self.forward_single(input_ids, attention_mask)
            tone_logits = self.tone_head(pooled)
            tone_loss_fct = nn.MSELoss()
            tone_loss = tone_loss_fct(tone_logits, kwargs["tone_labels"].float())
            total_loss += tone_loss
            outputs["tone_loss"] = tone_loss
            outputs["tone_logits"] = tone_logits

        # --- Single-Class Classification Task (e.g., Bias/Ideology) ---
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
            bias_loss_fct = nn.CrossEntropyLoss()
            bias_loss = bias_loss_fct(bias_logits, kwargs["bias_labels"])
            total_loss += bias_loss
            outputs["bias_loss"] = bias_loss
            outputs["bias_logits"] = bias_logits
        elif "labels" in kwargs and "input_ids" in kwargs and self.num_bias_classes > 0:
            raise ValueError(
                "num_bias_classes > 0 but no bias_labels or bias_input_ids found"
            )

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

            mlm_loss_fct = nn.CrossEntropyLoss()
            mlm_loss = mlm_loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1)
            )
            total_loss += mlm_loss
            outputs["mlm_loss"] = mlm_loss
            outputs["mlm_logits"] = prediction_scores

        outputs["loss"] = total_loss
        return outputs

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

    # Add this method to your MultiTaskRoberta class or create a subclass
    def forward_classification(self, input_ids, attention_mask=None, labels=None):
        # Get the pooled representation
        pooled = self.forward_single(input_ids, attention_mask)

        # Use the new bias_head for single-class classification
        logits = self.bias_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Return in the format expected by Trainer
        return {
            "loss": loss,
            "logits": logits,
        }

    def save_checkpoint(self, path):
        """Save model checkpoint with config info"""
        config = {
            "name": getattr(self, "name", "roberta-base"),
            "hidden_size": self.backbone.config.hidden_size,
            "vocab_size": self.backbone.config.vocab_size,
            "num_themes": self.theme_head.out_features,
            "num_tones": self.tone_head.out_features,
        }
        if self.num_bias_classes is not None:
            config["num_bias_classes"] = self.num_bias_classes

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, path)
