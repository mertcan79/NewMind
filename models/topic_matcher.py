"""
TOPIC MATCHER MODULE
Uses sentence embeddings to match opinions to their relevant topics.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class TopicMatcher:
    """
    MATCHES OPINIONS TO TOPICS USING SENTENCE EMBEDDINGS AND COSINE SIMILARITY
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.TOPIC_MATCHER_MODEL
        self.model = None
        self.topic_embeddings = None
        self.topic_ids = None
        self.topic_texts = None

    def load_model(self):
        """LOAD THE SENTENCE TRANSFORMER MODEL"""
        from sentence_transformers import SentenceTransformer

        print(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully")

    def encode_topics(self, topic_ids: List[str], topic_texts: List[str], batch_size: int = 32):
        """
        ENCODE ALL TOPICS AND STORE THEIR EMBEDDINGS
        This is done once and used for all matching operations.
        """
        if self.model is None:
            self.load_model()

        print(f"Encoding {len(topic_texts)} topics...")
        self.topic_ids = topic_ids
        self.topic_texts = topic_texts

        # ENCODE TOPICS IN BATCHES
        all_embeddings = []
        total_batches = (len(topic_texts) + batch_size - 1) // batch_size

        for i in range(0, len(topic_texts), batch_size):
            batch_end = min(i + batch_size, len(topic_texts))
            batch_texts = topic_texts[i:batch_end]
            batch_num = (i // batch_size) + 1

            print(f"  Encoding batch {batch_num}/{total_batches}...")

            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)

        self.topic_embeddings = np.vstack(all_embeddings)
        print(f"Topic embeddings shape: {self.topic_embeddings.shape}")

    def match_opinion(
        self,
        opinion_text: str,
        top_k: int = 1,
        threshold: float = None
    ) -> List[Dict]:
        """
        MATCH A SINGLE OPINION TO THE MOST RELEVANT TOPIC(S)

        Args:
            opinion_text: The opinion text to match
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold

        Returns:
            List of dicts with topic_id, similarity, and topic_text
        """
        if self.topic_embeddings is None:
            raise ValueError("Topics not encoded. Call encode_topics() first.")

        threshold = threshold or settings.SIMILARITY_THRESHOLD

        # ENCODE THE OPINION
        opinion_embedding = self.model.encode(
            opinion_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # COMPUTE COSINE SIMILARITIES (DOT PRODUCT SINCE NORMALIZED)
        similarities = np.dot(self.topic_embeddings, opinion_embedding)

        # GET TOP-K INDICES
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append({
                    "topic_id": self.topic_ids[idx],
                    "similarity": float(sim),
                    "topic_text": self.topic_texts[idx]
                })

        return results

    def match_opinions_batch(
        self,
        opinion_texts: List[str],
        top_k: int = 1,
        threshold: float = None,
        batch_size: int = 32
    ) -> List[List[Dict]]:
        """
        MATCH MULTIPLE OPINIONS TO TOPICS IN BATCH
        More efficient than calling match_opinion repeatedly.
        """
        if self.topic_embeddings is None:
            raise ValueError("Topics not encoded. Call encode_topics() first.")

        threshold = threshold or settings.SIMILARITY_THRESHOLD

        # ENCODE ALL OPINIONS IN BATCHES
        print(f"Encoding {len(opinion_texts)} opinions...")
        all_embeddings = []
        total_batches = (len(opinion_texts) + batch_size - 1) // batch_size

        for i in range(0, len(opinion_texts), batch_size):
            batch_end = min(i + batch_size, len(opinion_texts))
            batch_texts = opinion_texts[i:batch_end]
            batch_num = (i // batch_size) + 1

            print(f"  Encoding batch {batch_num}/{total_batches}...")

            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)

        opinion_embeddings = np.vstack(all_embeddings)

        # COMPUTE ALL SIMILARITIES AT ONCE
        similarities = np.dot(opinion_embeddings, self.topic_embeddings.T)

        results = []
        for i in range(len(opinion_texts)):
            sim_scores = similarities[i]
            top_indices = np.argsort(sim_scores)[::-1][:top_k]

            opinion_results = []
            for idx in top_indices:
                sim = sim_scores[idx]
                if sim >= threshold:
                    opinion_results.append({
                        "topic_id": self.topic_ids[idx],
                        "similarity": float(sim),
                        "topic_text": self.topic_texts[idx]
                    })

            results.append(opinion_results)

        return results

    def save(self, path: Path):
        """SAVE THE TOPIC EMBEDDINGS AND METADATA"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "topic_ids": self.topic_ids,
            "topic_texts": self.topic_texts,
            "topic_embeddings": self.topic_embeddings,
            "model_name": self.model_name
        }

        with open(path / "topic_matcher.pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"Saved topic matcher to {path}")

    def load(self, path: Path):
        """LOAD SAVED TOPIC EMBEDDINGS AND METADATA"""
        path = Path(path)

        with open(path / "topic_matcher.pkl", "rb") as f:
            data = pickle.load(f)

        self.topic_ids = data["topic_ids"]
        self.topic_texts = data["topic_texts"]
        self.topic_embeddings = data["topic_embeddings"]
        self.model_name = data["model_name"]

        # LOAD THE MODEL
        self.load_model()

        print(f"Loaded topic matcher from {path}")


def evaluate_topic_matching(
    matcher: TopicMatcher,
    opinion_texts: List[str],
    true_topic_ids: List[str],
    top_k: int = 1
) -> Dict:
    """
    EVALUATE TOPIC MATCHING PERFORMANCE USING ACCURACY AND F1

    Args:
        matcher: Trained TopicMatcher
        opinion_texts: List of opinion texts
        true_topic_ids: Ground truth topic IDs
        top_k: Consider match correct if true topic is in top-k predictions

    Returns:
        Dict with evaluation metrics
    """
    from sklearn.metrics import accuracy_score

    print(f"Evaluating on {len(opinion_texts)} opinions...")

    # GET PREDICTIONS
    predictions = matcher.match_opinions_batch(opinion_texts, top_k=top_k, threshold=0.0)

    # CHECK IF TRUE TOPIC IS IN PREDICTIONS
    correct = 0
    predicted_ids = []

    for i, (pred_list, true_id) in enumerate(zip(predictions, true_topic_ids)):
        pred_topic_ids = [p["topic_id"] for p in pred_list]

        # FOR ACCURACY: USE TOP-1 PREDICTION
        if pred_list:
            predicted_ids.append(pred_list[0]["topic_id"])
        else:
            predicted_ids.append(None)

        # FOR TOP-K ACCURACY
        if true_id in pred_topic_ids:
            correct += 1

    top_k_accuracy = correct / len(opinion_texts)

    # COMPUTE STANDARD METRICS (USING TOP-1 PREDICTION)
    accuracy = sum(1 for p, t in zip(predicted_ids, true_topic_ids) if p == t) / len(true_topic_ids)

    results = {
        "accuracy_top1": accuracy,
        f"accuracy_top{top_k}": top_k_accuracy,
        "total_samples": len(opinion_texts),
        "correct_predictions": correct
    }

    return results


if __name__ == "__main__":
    # TEST THE TOPIC MATCHER
    from data.preprocessing import DataProcessor

    # LOAD DATA
    processor = DataProcessor()
    processor.load_data()

    # PREPARE DATA SPLITS
    splits = processor.prepare_topic_matching_splits()

    # INITIALIZE MATCHER
    matcher = TopicMatcher()
    matcher.load_model()

    # ENCODE ALL TOPICS (USING TRAINING TOPICS FOR NOW)
    train_topics = splits["train"]["topics"]
    matcher.encode_topics(
        topic_ids=train_topics["topic_id"].tolist(),
        topic_texts=train_topics["clean_text"].tolist()
    )

    # EVALUATE ON VALIDATION SET (SMALL SAMPLE FOR SPEED)
    val_opinions = splits["val"]["opinions"]
    sample_size = min(500, len(val_opinions))
    sample = val_opinions.sample(n=sample_size, random_state=42)

    results = evaluate_topic_matching(
        matcher,
        opinion_texts=sample["clean_text"].tolist(),
        true_topic_ids=sample["topic_id"].tolist(),
        top_k=5
    )

    print("\nEvaluation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
