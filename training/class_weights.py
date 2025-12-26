# training/class_weights.py
from collections import Counter
from typing import Dict, List

import torch


def compute_class_weights(
    data: List[dict],
    label_map: Dict[str, int],
    beta: float = 0.999,   # closer to 1 => stronger effect for rare classes
    min_weight: float = 0.5,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Class-balanced loss weights using 'effective number of samples' idea.
    Good for imbalanced datasets (Counterclaim/Rebuttal).

    weight_c = (1 - beta) / (1 - beta^n_c)
    then normalized to mean=1 and clipped.
    """
    counts = Counter(item["label"] for item in data)
    weights = [0.0] * len(label_map)

    for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
        n = max(1, counts.get(label, 0))
        w = (1.0 - beta) / (1.0 - (beta ** n))
        weights[idx] = w

    w = torch.tensor(weights, dtype=torch.float32)
    # normalize to mean 1
    w = w / w.mean()
    # clip to avoid exploding gradients
    w = torch.clamp(w, min=min_weight, max=max_weight)
    return w
