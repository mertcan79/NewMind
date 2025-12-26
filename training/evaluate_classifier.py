# training/evaluate_classifier.py
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from training.data_utils import load_opinion_data, OpinionDataset, stratified_split, ID_TO_LABEL


DATA_PATH = "data/opinions.csv"
MODEL_PATH = "models/trained_classifier"
OUTPUT_PATH = "evaluation_results/classifier_metrics.json"

BASE_MODEL_FALLBACK = "distilbert-base-uncased"
MAX_LENGTH = 64
VAL_RATIO = 0.2
SEED = 42
BATCH_SIZE = 8


@torch.no_grad()
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_opinion_data(DATA_PATH)
    _, val_data = stratified_split(data, val_ratio=VAL_RATIO, seed=SEED)

    model_dir = Path(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir if model_dir.exists() else BASE_MODEL_FALLBACK)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    val_ds = OpinionDataset(val_data, tokenizer, max_length=MAX_LENGTH)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    label_ids = list(range(len(ID_TO_LABEL)))
    target_names = [ID_TO_LABEL[i] for i in label_ids]

    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)

    out = {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": target_names,
        "val_samples": len(all_labels),
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Evaluation saved to", OUTPUT_PATH)
    print("Macro F1:", report["macro avg"]["f1-score"])
    print("Accuracy:", report["accuracy"])


if __name__ == "__main__":
    evaluate()
