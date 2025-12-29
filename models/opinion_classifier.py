import os

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

LABEL_MAP = {
    0: "Claim",
    1: "Evidence",
    2: "Counterclaim",
    3: "Rebuttal",
}

MODEL_DIR = "models/trained_classifier"
BASE_MODEL = "distilbert-base-uncased"


class OpinionClassifier:
    def __init__(self, device=None, confidence_threshold=0.40):
        """
        Opinion classification model wrapper.

        Args:
            device (str): "cpu" or "cuda". Auto-detected if None.
            confidence_threshold (float): Minimum probability required
                                          to accept a prediction.
        """
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.confidence_threshold = confidence_threshold

        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """
        Load trained model if available, otherwise fallback to base model.
        """
        if os.path.exists(MODEL_DIR):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_DIR
            )
        else:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=len(LABEL_MAP)
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        """
        Predict opinion type for a single text.

        Returns:
            dict:
                {
                    "label": str,
                    "confidence": float,
                    "probabilities": {label: prob}
                }
        """
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=0)

        confidence, pred_id = torch.max(probs, dim=0)

        label = LABEL_MAP[pred_id.item()]
        confidence = confidence.item()

        probabilities = {
            LABEL_MAP[i]: round(probs[i].item(), 4)
            for i in range(len(LABEL_MAP))
        }

        if confidence < self.confidence_threshold:
            return {
                "label": "Uncertain",
                "confidence": confidence,
                "probabilities": probabilities,
            }

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def batch_predict(self, texts):
        """
        Predict opinion types for a batch of texts.

        Args:
            texts (List[str])

        Returns:
            List[dict]
        """
        if not texts:
            return []

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        probs = F.softmax(outputs.logits, dim=1)

        results = []

        for i in range(len(texts)):
            confidence, pred_id = torch.max(probs[i], dim=0)

            label = LABEL_MAP[pred_id.item()]
            confidence = confidence.item()

            prob_map = {
                LABEL_MAP[j]: round(probs[i][j].item(), 4)
                for j in range(len(LABEL_MAP))
            }

            if confidence < self.confidence_threshold:
                results.append({
                    "label": "Uncertain",
                    "confidence": confidence,
                    "probabilities": prob_map,
                })
            else:
                results.append({
                    "label": label,
                    "confidence": confidence,
                    "probabilities": prob_map,
                })

        return results

    def is_trained(self):
        """
        Check whether a fine-tuned model is loaded.
        """
        return os.path.exists(MODEL_DIR)
