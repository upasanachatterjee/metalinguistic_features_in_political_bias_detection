import torch
from typing import List, Dict, Any, Optional

class MultiLabelCollator:
    def __init__(self, top_themes_path: str, limit = 1000):
        with open(top_themes_path, "r") as f:
            self.top_themes = [line.strip() for line in f if line.strip()]
            self.top_themes = self.top_themes[:min(limit, len(self.top_themes))]
        self.theme_to_idx = {theme: idx for idx, theme in enumerate(self.top_themes)}
        self.num_labels = len(self.top_themes)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        themes = [item['V2Themes'] for item in batch]
        themes = [parse_multilabel(theme) for theme in themes]  # parse and dedupe

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Create multi-hot labels for each item in the batch
        batch_size = len(batch)
        multi_hot_labels = torch.zeros(batch_size, self.num_labels, dtype=torch.float)
        
        for batch_idx, theme_list in enumerate(themes):
            if theme_list is None:
                continue  # leave as all-zero vector
            for theme in theme_list:
                idx = self.theme_to_idx.get(theme)
                if idx is not None:
                    multi_hot_labels[batch_idx, idx] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": multi_hot_labels
        }
    

def parse_multilabel(theme_raw: Optional[str]) -> Optional[List[str]]:
        """Turn dataset label into a multi-hot list.
        Accepts list[int]/list[bool]/str with comma/space-delimited indices.
        """
        if theme_raw is None:
                # keep it None (leave empty)
                return None
        else:
            s_str = str(theme_raw).strip()
            if s_str == "":
                # non-None but empty string -> produce empty list (keeps it explicit)
                return None
            else:
                parts = [p for p in s_str.split(";") if p and p.strip()]
                seen = set()
                result = []
                for part in parts:
                    if ',' in part:
                        theme = part.rsplit(',', 1)[0].strip()
                    else:
                        theme = part.strip()
                    if theme and theme not in seen:
                        seen.add(theme)
                        result.append(theme)
                return result