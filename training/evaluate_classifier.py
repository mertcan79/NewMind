import json
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report

from training.data_utils import load_opinion_data, OpinionDataset, stratified_split, ID_TO_LABEL


DATA_PATH = "data/opinions.json"
MODEL_PATH = "models/trained_classifier"
OUTPUT_PATH = "evaluation_results/classifier_metrics.json"


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    raw_data = load_opinion_data(DATA_PATH)
    _, _, test_data = stratified_split(raw_data)

    test_dataset = OpinionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))],
        output_dict=True,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Evaluation saved to", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate()
