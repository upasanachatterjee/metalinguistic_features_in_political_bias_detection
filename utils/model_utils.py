from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from model import MultiTaskRoberta

BERT = "google-bert/bert-base-uncased"
BART = "facebook/bart-base"
BART_LARGE = "facebook/bart-large"
ROBERTA = "FacebookAI/roberta-base"
POLITICS = "launch/POLITICS"
IDEOLOGY_CLASSIFIER = "dragonslayer631/ideology_classifier_finetuned"


def load_model(model):
    if model in [BERT, BART, ROBERTA, POLITICS]:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)
        return tokenizer, model
    else:
        print("Attempting to load as MultiTaskRoberta...")
        try:
            themes = 1000 if "1000" in model else 2000
            tones = 1 if "tone_tone" in model else 2
            print(f"loading w num_themes={themes}, num_tones={tones}")

            checkpoint = torch.load(model, map_location="cpu")
            # Format A: checkpoint dict with "model_state_dict" key
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                # Format B: bare state dict
                state = checkpoint

            classification_model = MultiTaskRoberta(
                num_tones=tones, num_themes=themes, num_bias_classes=3
            )
            classification_model.load_state_dict(state, strict=False)

            # Get the tokenizer from the base model name
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")

            print(f"Successfully loaded MultiTaskRoberta from {model}")
            print("Using tokenizer from roberta-base")

            return tokenizer, classification_model

        except Exception as e:
            print(f"Failed to load as MultiTaskRoberta: {e}")
            print("Falling back to legacy custom loading...")

            # Fallback to original custom loading logic
            state = torch.load(f"{model}/pytorch_model.bin", map_location="cpu")
            config = AutoConfig.from_pretrained("roberta-base", num_labels=3)
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", config=config
            )
            model.load_state_dict(
                state, strict=False
            )  # strict=False tolerates head mismatches
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            return tokenizer, model


def get_model_name(model) -> str:
    if model == BERT:
        return "bert"
    elif model == BART:
        return "bart"
    elif model == ROBERTA:
        return "roberta"
    elif model == POLITICS:
        return "politics"
    else:
        return "custom"
