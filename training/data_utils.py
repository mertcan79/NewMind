import csv
import random
from collections import Counter

LABEL_MAP = {
    "Claim": 0,
    "Evidence": 1,
    "Counterclaim": 2,
    "Rebuttal": 3,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_opinion_data(data_path):
    """
    Load opinion data from CSV.

    Expected CSV columns:
    - text OR opinion OR opinion_text
    - label OR type
    """
    data = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = (
                row.get("text")
                or row.get("opinion")
                or row.get("opinion_text")
            )

            label = row.get("label") or row.get("type")

            if not text or not label:
                continue

            label = label.strip()

            if label not in LABEL_MAP:
                continue

            data.append({
                "text": text.strip(),
                "label": label,
            })

    if not data:
        raise ValueError("No valid opinion samples loaded from CSV.")

    return data


def stratified_split(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    buckets = {}
    for item in data:
        buckets.setdefault(item["label"], []).append(item)

    train, val, test = [], [], []

    for label, items in buckets.items():
        random.shuffle(items)
        n = len(items)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train.extend(items[:train_end])
        val.extend(items[train_end:val_end])
        test.extend(items[val_end:])

    return train, val, test


class OpinionDataset:
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": LABEL_MAP[item["label"]],
        }
