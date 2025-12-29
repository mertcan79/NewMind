# training/train_classifier.py
import os
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from training.data_utils import load_opinion_data, stratified_split, OpinionDataset, LABEL_MAP, ID_TO_LABEL
from training.class_weights import compute_class_weights


# CONFIGURATION
DATA_PATH = "data/opinions.csv"
MODEL_OUTPUT_DIR = "models/trained_classifier"
BASE_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
PATIENCE = 2
VAL_RATIO = 0.15
SEED = 42

WARMUP_RATIO = 0.1
WEIGHT_BETA = 0.9999  # Higher value gives more weight to minority classes 


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(model, dataloader, device, verbose=False):
    """Evaluate model and return metrics"""
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

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted"))

    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(classification_report(all_labels, all_preds,
                                    target_names=[ID_TO_LABEL[i] for i in range(len(LABEL_MAP))],
                                    digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("="*60 + "\n")

    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def train():
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LOAD DATA
    print(f"\nLoading data from {DATA_PATH}...")
    data = load_opinion_data(DATA_PATH)
    print(f"Total samples: {len(data)}")

    # Display class distribution
    from collections import Counter
    label_counts = Counter(item['label'] for item in data)
    print("\nClass distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15s}: {count:6d} ({100*count/len(data):5.2f}%)")

    # SPLIT DATA
    train_data, val_data = stratified_split(data, val_ratio=VAL_RATIO, seed=SEED)
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")

    # PREPARE DATASETS
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    train_ds = OpinionDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_ds = OpinionDataset(val_data, tokenizer, max_length=MAX_LENGTH)

    # CREATE WEIGHTED SAMPLER TO HANDLE CLASS IMBALANCE
    print("\nCreating weighted sampler for class balance...")
    labels = [train_ds[i]["labels"].item() for i in range(len(train_ds))]
    class_counts = np.bincount(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # INITIALIZE MODEL
    print(f"\nInitializing model: {BASE_MODEL}")
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABEL_MAP)
    ).to(device)

    # COMPUTE CLASS WEIGHTS FOR LOSS FUNCTION
    class_weights = compute_class_weights(
        train_data, LABEL_MAP, beta=WEIGHT_BETA
    ).to(device)
    print(f"\nClass weights (beta={WEIGHT_BETA}):")
    for i, (label, _) in enumerate(sorted(LABEL_MAP.items(), key=lambda x: x[1])):
        print(f"  {label:15s}: {class_weights[i]:.4f}")

    loss_fn = CrossEntropyLoss(weight=class_weights)

    # OPTIMIZER AND SCHEDULER
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Max sequence length: {MAX_LENGTH}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    best_macro_f1 = -1.0
    best_weighted_f1 = -1.0
    epochs_no_improve = 0

    out_dir = Path(MODEL_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TRAINING LOOP
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        num_batches = len(train_loader)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)

        for step, batch in enumerate(train_loader):
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

            # Log progress every 100 steps
            if (step + 1) % 100 == 0 or (step + 1) == num_batches:
                elapsed = time.time() - start_time
                steps_done = step + 1
                avg_loss = running_loss / steps_done

                if steps_done < num_batches:
                    steps_left = num_batches - steps_done
                    sec_per_step = elapsed / steps_done
                    eta_min = (steps_left * sec_per_step) / 60
                    print(f"  Step {steps_done:4d}/{num_batches} | Loss: {avg_loss:.4f} | ETA: {eta_min:.1f}min")
                else:
                    print(f"  Step {steps_done:4d}/{num_batches} | Loss: {avg_loss:.4f} | Done!")

        # VALIDATION
        avg_train_loss = running_loss / num_batches
        print(f"\nValidating...")
        val_metrics = evaluate_model(model, val_loader, device, verbose=False)

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss:    {avg_train_loss:.4f}")
        print(f"  Val Macro F1:  {val_metrics['macro_f1']:.4f}")
        print(f"  Val Weighted F1: {val_metrics['weighted_f1']:.4f}")

        # SAVE BEST MODEL (based on macro F1 to ensure all classes are learned)
        if val_metrics['macro_f1'] > best_macro_f1 + 1e-4:
            best_macro_f1 = val_metrics['macro_f1']
            best_weighted_f1 = val_metrics['weighted_f1']
            epochs_no_improve = 0

            print(f"  *** New best model! Saving to {out_dir}")
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

            # Save detailed metrics
            with open(out_dir / "train_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": BASE_MODEL,
                        "max_length": MAX_LENGTH,
                        "epochs": epoch,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                        "patience": PATIENCE,
                        "best_val_macro_f1": best_macro_f1,
                        "best_val_weighted_f1": best_weighted_f1,
                        "class_weights": class_weights.detach().cpu().tolist(),
                    },
                    f,
                    indent=2,
                )
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= PATIENCE:
                print(f"\n*** Early stopping triggered (no improvement for {PATIENCE} epochs)")
                break

    # FINAL EVALUATION
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Macro F1:    {best_macro_f1:.4f}")
    print(f"Best Weighted F1: {best_weighted_f1:.4f}")

    # Load best model and evaluate on validation set with details
    print("\nLoading best model for detailed evaluation...")
    model = DistilBertForSequenceClassification.from_pretrained(out_dir).to(device)
    print("\nValidation Set Performance:")
    evaluate_model(model, val_loader, device, verbose=True)

    print(f"\nModel saved to: {out_dir}")
    return best_macro_f1


if __name__ == "__main__":
    train()
