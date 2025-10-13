import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, PreTrainedModel, RobertaConfig
import torch


class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class MultiTaskRoberta(nn.Module):
    def __init__(self, name="roberta-base", emb_dim=256, theme_path="top_themes.txt"):
        super().__init__()
        # Single backbone model
        self.backbone = AutoModel.from_pretrained(name)
        self.pool = MeanPooler()
        hid = self.backbone.config.hidden_size

        # Task-specific heads
        self.proj = nn.Sequential(nn.Linear(hid, emb_dim), nn.GELU())
        num_themes = 0

        with open(theme_path) as f:
            num_themes = len(f.readlines())
        self.theme_head = nn.Linear(hid, num_themes)
        self.tone_head = nn.Linear(hid, 2)

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

    def forward(self, input_ids=None, attention_mask=None, mlm=False, labels=None, **kwargs):
        # Clone inputs to prevent in-place modifications affecting autograd
        if input_ids is not None:
            input_ids = input_ids.clone()
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
            
        # Use the single backbone for all tasks
        backbone_output = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        if mlm:
            # For MLM, apply the LM head to the last hidden states
            sequence_output = backbone_output.last_hidden_state
            prediction_scores = self.lm_head(sequence_output)
            
            masked_lm_loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, prediction_scores.size(-1)), 
                    labels.view(-1)
                )
            
            # Return in the format expected by the training loop
            return type('MLMOutput', (), {
                'loss': masked_lm_loss,
                'logits': prediction_scores
            })()
        else:
            # For other tasks, pool and return embeddings
            pooled = self.pool(backbone_output.last_hidden_state, attention_mask)
            z = F.normalize(self.proj(pooled), p=2, dim=-1)
            return z, pooled  # z for triplet, pooled for themes/tone

    # Add this method to your MultiTaskRoberta class or create a subclass
    def forward_classification(self, input_ids, attention_mask=None, labels=None):
        # Get the pooled representation
        _, pooled = super().forward(input_ids, attention_mask)
        
        # Use either the existing theme_head or a new bias_head
        # Option 1: Using theme_head (if repurposed)
        logits = self.theme_head(pooled)
        
        # Option 2: Using a new bias_head (if added)
        # logits = self.bias_head(pooled)
        
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
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "name": getattr(self, "name", "roberta-base"),
                "emb_dim": self.proj[0].out_features,
                "hidden_size": self.backbone.config.hidden_size,
                "vocab_size": self.backbone.config.vocab_size,
            },
        }
        torch.save(checkpoint, path)


class BiasClassifier(PreTrainedModel):
    config_class = RobertaConfig

    def __init__(
        self, config, pretrained_path: str, num_bias_classes=3, freeze_backbone=False
    ):
        super().__init__(config)

        if pretrained_path:
            # Load pretrained checkpoint
            with open(pretrained_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")

            # Initialize backbone and pooler (reuse from pretrained)
            self.backbone = AutoModel.from_config(config)
            self.pool = MeanPooler()

            # Load pretrained backbone weights
            pretrained_state = checkpoint["model_state_dict"]
            backbone_state = {
                k.replace("backbone.", ""): v
                for k, v in pretrained_state.items()
                if k.startswith("backbone.")
            }
            self.backbone.load_state_dict(backbone_state)
        else:
            raise ValueError("Pretrained path is required")

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # New bias classification head
        self.bias_head = nn.Linear(config.hidden_size, num_bias_classes)
        self.num_labels = num_bias_classes

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        logits = self.bias_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Return format expected by Trainer
        return {
            "loss": loss,
            "logits": logits,
        }

    @classmethod
    def from_pretrained_checkpoint(
        cls, pretrained_path, num_bias_classes=3, freeze_backbone=False
    ):
        """Create model from your custom checkpoint"""
        config = RobertaConfig.from_pretrained("roberta-base")
        config.num_labels = num_bias_classes
        return cls(config, pretrained_path, num_bias_classes, freeze_backbone)
