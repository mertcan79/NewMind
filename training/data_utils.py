# training/data_utils.py
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

LABEL_MAP: Dict[str, int] = {
    "Claim": 0,
    "Evidence": 1,
    "Counterclaim": 2,
    "Rebuttal": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def _safe_strip(x: Optional[str]) -> str:
    return (x or "").strip()


def load_opinion_data(data_path: str) -> List[Dict]:
    """
    Loads opinions from CSV.
    Expected columns (case-insensitive):
      - text (or opinion / opinion_text)
      - label (or position)
    Ignores blank lines and rows with empty text/label.
    """
    data: List[Dict] = []
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {data_path}")

        # normalize header names
        field_map = {name.lower().strip(): name for name in reader.fieldnames}

        def get_col(*candidates: str) -> Optional[str]:
            for c in candidates:
                if c in field_map:
                    return field_map[c]
            return None

        text_col = get_col("text", "opinion", "opinion_text")
        label_col = get_col("label", "position", "type")

        if not text_col or not label_col:
            raise ValueError(
                f"CSV must include text+label columns. Found headers: {reader.fieldnames}"
            )

        for row in reader:
            text = _safe_strip(row.get(text_col))
            label = _safe_strip(row.get(label_col))

            if not text or not label:
                continue
            if label not in LABEL_MAP:
                # skip unknown labels instead of crashing
                continue

            data.append({"text": text, "label": label})

    if not data:
        raise ValueError(f"No valid rows loaded from {data_path}")

    return data


def stratified_split(
    data: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Deterministic stratified split by label.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)

    by_label = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class OpinionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL_MAP[item["label"]], dtype=torch.long),
        }
