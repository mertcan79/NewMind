# training/train_classifier.py
import os
import json
import random
from pathlib import Path

from tqdm.auto import tqdm
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.metrics import f1_score

from training.data_utils import load_opinion_data, stratified_split, OpinionDataset, LABEL_MAP
from training.class_weights import compute_class_weights


DATA_PATH = "data/opinions.csv"
from pathlib import Path

MODEL_OUTPUT_DIR = "models/trained_classifier"
BASE_MODEL = "distilbert-base-uncased"

model_path = Path(MODEL_OUTPUT_DIR)
if model_path.exists() and (model_path / "config.json").exists():
    print(f"Loading existing checkpoint from {MODEL_OUTPUT_DIR}")
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=len(LABEL_MAP))
else:
    print(f"Loading base model {BASE_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABEL_MAP))

MAX_LENGTH = 64

EPOCHS = 3
BATCH_SIZE = 8
LR = 5e-4
PATIENCE = 1
VAL_RATIO = 0.2
SEED = 42

WARMUP_RATIO = 0.06
WEIGHT_BETA = 0.9999 


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_macro_f1(model, dataloader, device) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return float(f1_score(all_labels, all_preds, average="macro"))


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_opinion_data(DATA_PATH)
    train_data, val_data = stratified_split(data, val_ratio=VAL_RATIO, seed=SEED)

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    train_ds = OpinionDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_ds = OpinionDataset(val_data, tokenizer, max_length=MAX_LENGTH)
    from torch.utils.data import WeightedRandomSampler

    # after train_ds is created:
    labels = [train_ds[i]["labels"].item() for i in range(len(train_ds))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABEL_MAP)
    ).to(device)

    # ✅ class-weighted loss (computed from TRAIN only)
    class_weights = compute_class_weights(
        train_data, LABEL_MAP, beta=WEIGHT_BETA
    ).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = -1.0
    epochs_no_improve = 0

    out_dir = Path(MODEL_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        start_time = time.time()
        num_batches = len(train_loader)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Optional: log every N steps
            if step > 0 and step % 200 == 0:
                elapsed = time.time() - start_time
                steps_done = step + 1
                steps_left = num_batches - steps_done
                sec_per_step = elapsed / steps_done
                eta_min = (steps_left * sec_per_step) / 60

                tqdm.write(
                    f"  step {steps_done}/{num_batches} | "
                    f"loss={loss.item():.4f} | ETA ~{eta_min:.1f} min"
                )


        val_f1 = eval_macro_f1(model, val_loader, device)
        avg_loss = running_loss / max(1, len(train_loader))

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

        # Early stopping on macro-F1
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            epochs_no_improve = 0

            # ✅ save best
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

            with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": BASE_MODEL,
                        "max_length": MAX_LENGTH,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "patience": PATIENCE,
                        "best_val_macro_f1": best_f1,
                        "class_weights": class_weights.detach().cpu().tolist(),
                    },
                    f,
                    indent=2,
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping (no improvement for {PATIENCE} epochs). Best macro-F1={best_f1:.4f}")
                break

    print("Training complete. Best val macro-F1:", best_f1)


if __name__ == "__main__":
    train()
