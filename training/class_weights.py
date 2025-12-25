from collections import Counter
import torch


def compute_class_weights(data, label_map):
    counts = Counter(item["label"] for item in data)
    total = sum(counts.values())

    weights = []
    for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
        weights.append(total / (len(counts) * counts[label]))

    return torch.tensor(weights, dtype=torch.float)
