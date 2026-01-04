import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List


class RegressionCollator:
    def __init__(self, output_size: int = 1):
        self.output_size = output_size

    def __call__(self, batch):
        # Filter out samples with invalid regression values
        valid_samples = []
        for item in batch:
            tone = item.get("V2Tone")
            parsed_values = parse_regression_values(
                str(tone) if tone is not None else None, self.output_size
            )
            if parsed_values is not None:
                valid_samples.append((item, parsed_values))

        # If no valid samples, return empty batch
        if not valid_samples:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "targets": torch.empty((0, self.output_size), dtype=torch.float),
                "_skip": True,  # Indicate to skip this batch
            }

        # Extract data from valid samples only
        input_ids = [
            torch.tensor(item["input_ids"], dtype=torch.long)
            for item, _ in valid_samples
        ]
        attention_mask = [
            torch.tensor(item["attention_mask"], dtype=torch.long)
            for item, _ in valid_samples
        ]
        parsed_values = [
            torch.tensor(parsed_value, dtype=torch.float)
            for _, parsed_value in valid_samples
        ]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "targets": torch.stack(parsed_values),
        }


def parse_regression_values(
    row_string: Optional[str], output_size: int = 2
) -> Optional[List[float]]:
    """
    Parse comma-separated string and return the first output_size values as floats.
    If fewer values exist, pad with zeros.

    Args:
        row_string: Comma-separated string of float values
        output_size: Number of values to return

    Returns:
        List of float values, or None if parsing fails
    """
    if row_string is None:
        return None

    try:
        values = row_string.split(",")
        if not values:
            return None

        # Parse available values
        result = []
        for i in range(min(len(values), output_size)):
            result.append(float(values[i]))

        # Pad with zeros if needed
        while len(result) < output_size:
            result.append(0.0)

        return result
    except (ValueError, AttributeError):
        return None
