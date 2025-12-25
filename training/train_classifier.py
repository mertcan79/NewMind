import os
import json
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW


from training.data_utils import load_opinion_data, stratified_split, OpinionDataset, LABEL_MAP
from training.class_weights import compute_class_weights


DATA_PATH = "data/opinions.csv"
MODEL_OUTPUT_DIR = "models/trained_classifier"
EPOCHS = 4
BATCH_SIZE = 16
LR = 2e-5
PATIENCE = 2


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    raw_data = load_opinion_data(DATA_PATH)
    train_data, val_data, test_data = stratified_split(raw_data)

    train_dataset = OpinionDataset(train_data, tokenizer)
    val_dataset = OpinionDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABEL_MAP)
    ).to(device)

    class_weights = compute_class_weights(train_data, LABEL_MAP).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    print("Training completed.")


def evaluate(model, dataloader, device):
    from sklearn.metrics import f1_score

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
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

    return f1_score(all_labels, all_preds, average="macro")


if __name__ == "__main__":
    train()
