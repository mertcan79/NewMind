"""
OPINION CLASSIFIER MODULE
Fine-tunes DistilBERT for classifying opinion types.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, OPINION_TYPES, OPINION_TYPE_TO_ID, ID_TO_OPINION_TYPE


class OpinionDataset(Dataset):
    """PYTORCH DATASET FOR OPINION CLASSIFICATION"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: DistilBertTokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


class OpinionClassifier:
    """
    DISTILBERT-BASED CLASSIFIER FOR OPINION TYPES
    Classifies into: Claim, Counterclaim, Rebuttal, Evidence
    """

    def __init__(self, model_name: str = None, num_labels: int = 4):
        self.model_name = model_name or settings.CLASSIFIER_MODEL
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None

    def load_model(self, pretrained_path: Optional[Path] = None):
        """LOAD TOKENIZER AND MODEL"""

        print(f"Using device: {self.device}")

        if pretrained_path and Path(pretrained_path).exists():
            print(f"Loading fine-tuned model from {pretrained_path}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_path)
        else:
            print(f"Loading base model: {self.model_name}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )

        self.model.to(self.device)
        print("Model loaded successfully")

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        FINE-TUNE THE MODEL ON TRAINING DATA

        Returns:
            Training history with metrics
        """
        epochs = epochs or settings.NUM_EPOCHS
        batch_size = batch_size or settings.TRAIN_BATCH_SIZE
        learning_rate = learning_rate or settings.LEARNING_RATE

        if self.model is None:
            self.load_model()

        # CREATE DATASETS
        train_dataset = OpinionDataset(
            train_texts, train_labels, self.tokenizer, settings.MAX_SEQ_LENGTH
        )
        val_dataset = OpinionDataset(
            val_texts, val_labels, self.tokenizer, settings.MAX_SEQ_LENGTH
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=settings.EVAL_BATCH_SIZE)

        # OPTIMIZER AND SCHEDULER
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}
        best_f1 = 0

        # TRAINING LOOP
        for epoch in range(epochs):
            # TRAINING PHASE
            self.model.train()
            train_loss = 0
            total_batches = len(train_loader)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # PRINT PROGRESS EVERY 50 BATCHES
                if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                    print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {loss.item():.4f}")

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # VALIDATION PHASE
            val_metrics = self.evaluate(val_texts, val_labels, batch_size=settings.EVAL_BATCH_SIZE)
            history["val_loss"].append(val_metrics.get("loss", 0))
            history["val_f1"].append(val_metrics["f1_weighted"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(f"\n  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

            # SAVE BEST MODEL
            if val_metrics["f1_weighted"] > best_f1 and save_path:
                best_f1 = val_metrics["f1_weighted"]
                self.save(save_path)
                print(f"  Saved best model (F1: {best_f1:.4f})")

        return history

    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """PREDICT OPINION TYPES FOR TEXTS"""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()
        predictions = []

        dataset = OpinionDataset(
            texts,
            [0] * len(texts),  # DUMMY LABELS
            self.tokenizer,
            settings.MAX_SEQ_LENGTH
        )
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy().tolist())

        return predictions

    def predict_with_labels(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """PREDICT AND RETURN STRING LABELS"""
        predictions = self.predict(texts, batch_size)
        return [ID_TO_OPINION_TYPE[p] for p in predictions]

    def predict_single(self, text: str) -> Dict:
        """PREDICT FOR A SINGLE TEXT WITH PROBABILITIES"""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=settings.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.softmax(outputs.logits, dim=1).squeeze()
            pred_idx = torch.argmax(probs).item()

        return {
            "predicted_type": ID_TO_OPINION_TYPE[pred_idx],
            "predicted_id": pred_idx,
            "probabilities": {
                ID_TO_OPINION_TYPE[i]: float(probs[i])
                for i in range(len(OPINION_TYPES))
            }
        }

    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 32
    ) -> Dict:
        """
        EVALUATE THE MODEL ON TEST DATA

        Returns:
            Dict with accuracy, F1 scores, and classification report
        """
        predictions = self.predict(texts, batch_size)

        accuracy = accuracy_score(labels, predictions)
        f1_weighted = f1_score(labels, predictions, average="weighted")
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_per_class = f1_score(labels, predictions, average=None)

        report = classification_report(
            labels,
            predictions,
            target_names=OPINION_TYPES,
            output_dict=True
        )

        conf_matrix = confusion_matrix(labels, predictions)

        return {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "f1_per_class": {
                OPINION_TYPES[i]: float(f1_per_class[i])
                for i in range(len(OPINION_TYPES))
            },
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }

    def save(self, path: Path):
        """SAVE THE FINE-TUNED MODEL"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f"Model saved to {path}")

    def load(self, path: Path):
        """LOAD A FINE-TUNED MODEL"""
        self.load_model(pretrained_path=path)


if __name__ == "__main__":
    # TEST THE CLASSIFIER
    from data.preprocessing import DataProcessor

    # LOAD DATA
    processor = DataProcessor()
    processor.load_data()

    # GET CLASSIFICATION DATA
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        processor.prepare_classification_data()

    # INITIALIZE CLASSIFIER
    classifier = OpinionClassifier()
    classifier.load_model()

    # TRAIN (USE SMALL SAMPLE FOR TESTING)
    sample_size = 500
    history = classifier.train(
        train_texts=train_texts[:sample_size],
        train_labels=train_labels[:sample_size],
        val_texts=val_texts[:100],
        val_labels=val_labels[:100],
        epochs=1,
        save_path=Path("trained_models/classifier")
    )

    # EVALUATE
    print("\nEvaluating on test sample...")
    results = classifier.evaluate(
        texts=test_texts[:200],
        labels=test_labels[:200]
    )

    print("\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print("\n  Per-class F1:")
    for label, f1 in results["f1_per_class"].items():
        print(f"    {label}: {f1:.4f}")
