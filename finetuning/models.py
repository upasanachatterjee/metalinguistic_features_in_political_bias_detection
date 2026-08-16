import os

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from model import MultiTaskRoberta

BERT = "google-bert/bert-base-uncased"
BART = "facebook/bart-base"
ROBERTA = "FacebookAI/roberta-base"
POLITICS = "launch/POLITICS"
IDEOLOGY_CLASSIFIER = "upasanachatterjee/ideology_classifier_finetuned"

HUB_BASELINES = (BERT, BART, ROBERTA, POLITICS)

# The bias task is left / center / right.
NUM_BIAS_CLASSES = 3
# Pretraining always used the full top_themes.txt label space.
NUM_THEMES = 2000


def load_model(model_ref: str):
    """Load one of the study's models, ready for bias classification.

    `model_ref` is either a hub name from HUB_BASELINES, a path to a
    MultiTaskRoberta `.pt` checkpoint, or a directory holding a legacy
    `pytorch_model.bin`. Returns (tokenizer, model); the tokenizer is always
    roberta-base's for the checkpoint paths, since that is the backbone they
    were pretrained on.
    """
    if model_ref in HUB_BASELINES:
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ref, num_labels=NUM_BIAS_CLASSES
        )
        return tokenizer, model

    if os.path.isdir(model_ref):
        return _load_legacy_directory(model_ref)

    return _load_multitask_checkpoint(model_ref)


def _load_multitask_checkpoint(path: str):
    """Rebuild MultiTaskRoberta from a pretraining checkpoint, plus a bias head.

    The bias head is new (pretraining never trains one), so it starts random --
    hence `strict=False`. The tone head size is read back from the weights
    because it has changed across runs.
    """
    checkpoint = torch.load(path, map_location="cpu")
    # Either a full checkpoint dict from MultiTaskRoberta.save_checkpoint, or a
    # bare state dict from an older run.
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    num_tones = (
        state["tone_head.weight"].shape[0] if "tone_head.weight" in state else 1
    )
    print(f"loading MultiTaskRoberta from {path} (themes={NUM_THEMES}, tones={num_tones})")

    model = MultiTaskRoberta(
        num_tones=num_tones,
        num_themes=NUM_THEMES,
        num_bias_classes=NUM_BIAS_CLASSES,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  {len(missing)} missing / {len(unexpected)} unexpected keys")

    return AutoTokenizer.from_pretrained("roberta-base"), model


def _load_legacy_directory(path: str):
    """Load a pre-MultiTaskRoberta run: a directory with a `pytorch_model.bin`.

    These were plain roberta-base sequence classifiers, so the multi-task heads
    in the state dict (if any) are dropped by `strict=False`.
    """
    print(f"loading legacy sequence classifier from {path}")
    state = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
    config = AutoConfig.from_pretrained("roberta-base", num_labels=NUM_BIAS_CLASSES)
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )
    model.load_state_dict(state, strict=False)
    return AutoTokenizer.from_pretrained("roberta-base"), model


def get_model_name(model_ref: str) -> str:
    """Short label used in output directories and result filenames."""
    return {
        BERT: "bert",
        BART: "bart",
        ROBERTA: "roberta",
        POLITICS: "politics",
    }.get(model_ref, "custom")
