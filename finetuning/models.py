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
# MITweet's relevance task is one binary label per facet.
NUM_RELEVANCE_FACETS = 12
# Both BASIL tasks ask whether a sentence carries a bias span of one type.
NUM_BASIL_CLASSES = 2
# Pretraining always used the full top_themes.txt label space.
NUM_THEMES = 2000

# Every 3-class fine-tuning task reuses the bias head: MITweet ideology is 0=left/1=center/
# 2=right, and SemEval stance and opinion are 3 exclusive classes over the same 768->3 shape.
BIAS_TASKS = ("bias", "ideology", "stance", "opinion")
# BASIL's two tasks ride the same keys, but the head is 2-wide rather than 3.
BASIL_TASKS = ("lexical", "informational")


def load_model(model_ref: str, task: str = "bias"):
    """Load a hub name, a MultiTaskRoberta `.pt`, or a legacy `pytorch_model.bin`
    directory, with the head `task` needs. Checkpoints always get roberta-base's tokenizer.
    """
    if task not in BIAS_TASKS + BASIL_TASKS and task != "relevance":
        raise ValueError(
            f"unknown task {task!r}; expected one of "
            f"{BIAS_TASKS + BASIL_TASKS + ('relevance',)}"
        )

    if model_ref in HUB_BASELINES:
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        if task == "relevance":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_ref,
                num_labels=NUM_RELEVANCE_FACETS,
                problem_type="multi_label_classification",
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_ref, num_labels=_num_classes(task)
            )
        return tokenizer, model

    if os.path.isdir(model_ref):
        return _load_legacy_directory(model_ref, task)

    return _load_multitask_checkpoint(model_ref, task)


def _num_classes(task: str) -> int:
    """The classification head's width for one task."""
    if task == "relevance":
        return NUM_RELEVANCE_FACETS
    return NUM_BASIL_CLASSES if task in BASIL_TASKS else NUM_BIAS_CLASSES


def _head_sizes(task: str) -> dict:
    """The `MultiTaskRoberta` head arguments one task needs; the other head stays absent."""
    if task == "relevance":
        return {"num_relevance_facets": NUM_RELEVANCE_FACETS}
    return {"num_bias_classes": _num_classes(task)}


def _load_multitask_checkpoint(path: str, task: str = "bias"):
    """Rebuild MultiTaskRoberta from a pretraining or fine-tuning checkpoint, plus the head
    `task` needs.
    """
    # The task head may be new (`strict=False`); tone size is read back from the weights.
    checkpoint = torch.load(path, map_location="cpu")
    # A save_checkpoint dict, or a bare state dict from an older run.
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    num_tones = (
        state["tone_head.weight"].shape[0] if "tone_head.weight" in state else 1
    )
    print(f"loading MultiTaskRoberta from {path} (themes={NUM_THEMES}, tones={num_tones}, task={task})")

    model = MultiTaskRoberta(
        num_tones=num_tones,
        num_themes=NUM_THEMES,
        **_head_sizes(task),
    )
    # strict=False forgives missing and unexpected keys but still raises on a size mismatch,
    # which a 3-class fine-tuned bias head would hit against BASIL's 2-class one.
    expected = model.state_dict()
    mismatched = [
        key for key, value in state.items()
        if key in expected and value.shape != expected[key].shape
    ]
    for key in mismatched:
        del state[key]
    if mismatched:
        print(f"  dropped {len(mismatched)} head tensors sized for another task: {mismatched}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  {len(missing)} missing / {len(unexpected)} unexpected keys")

    return AutoTokenizer.from_pretrained("roberta-base"), model


def _load_legacy_directory(path: str, task: str = "bias"):
    """Load a pre-MultiTaskRoberta run: a directory with a `pytorch_model.bin`.

    These were plain roberta-base sequence classifiers, so the multi-task heads
    in the state dict (if any) are dropped by `strict=False`.
    """
    print(f"loading legacy sequence classifier from {path}")
    state = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
    # strict=False would drop every `backbone.*` key here and hand back untrained weights.
    if any(key.startswith("backbone.") for key in state):
        raise ValueError(
            f"{path}/pytorch_model.bin is a MultiTaskRoberta state dict, not a sequence "
            "classifier. Loading it here would silently discard every weight. Save the run "
            "with MultiTaskRoberta.save_checkpoint and pass the resulting .pt file instead."
        )
    num_labels = _num_classes(task)
    config = AutoConfig.from_pretrained("roberta-base", num_labels=num_labels)
    if task == "relevance":
        config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", config=config
    )
    # A head sized for another task can't be reused; drop it rather than fail the whole load.
    expected = model.state_dict()
    mismatched = [
        key for key, value in state.items()
        if key in expected and value.shape != expected[key].shape
    ]
    for key in mismatched:
        del state[key]
    if mismatched:
        print(f"  dropped {len(mismatched)} head tensors sized for another task: {mismatched}")
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
