"""
Computes the full opinion↔topic similarity matrix once, then slices per topic
to retrieve the most relevant opinions.

Key ideas:
- Encode topics once
- Encode opinions once
- Compute similarity matrix once: sims = opinion_emb @ topic_emb.T  (cosine if normalized)
- For each topic, slice sims[:, topic_idx] to rank opinions
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Keep same repo-style import behavior
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings  # noqa: E402


@dataclass
class Match:
    opinion_id: str
    opinion_text: str
    similarity: float


class TopicMatcher:
    """
    Topic ↔ Opinion matcher using sentence embeddings + cosine similarity.

    Workflow:
      1) matcher = TopicMatcher()
      2) matcher.encode_topics(topic_ids, topic_texts)
      3) matcher.encode_opinions(opinion_ids, opinion_texts)
      4) matcher.compute_similarity_matrix()
      5) matcher.top_opinions_for_topic(topic_id, top_k=10)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(settings, "EMBEDDING_MODEL_NAME", None) or getattr(
            settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.model = None

        # Topics
        self.topic_ids: List[str] = []
        self.topic_texts: List[str] = []
        self.topic_embeddings: Optional[np.ndarray] = None  # shape: (n_topics, dim)

        # Opinions
        self.opinion_ids: List[str] = []
        self.opinion_texts: List[str] = []
        self.opinion_embeddings: Optional[np.ndarray] = None  # shape: (n_opinions, dim)

        # Similarity matrix
        self.similarity_matrix: Optional[np.ndarray] = None  # shape: (n_opinions, n_topics)

        # Convenience maps
        self._topic_id_to_idx: Dict[str, int] = {}
        self._opinion_id_to_idx: Dict[str, int] = {}

    # ----------------------------
    # Model loading / encoding
    # ----------------------------
    def load_model(self):
        if self.model is not None:
            return

        from sentence_transformers import (
            SentenceTransformer,  # local import to keep deps optional
        )

        self.model = SentenceTransformer(self.model_name)

    def encode_topics(self, topic_ids: List[str], topic_texts: List[str], batch_size: int = 64):
        if len(topic_ids) != len(topic_texts):
            raise ValueError("topic_ids and topic_texts must have the same length")

        self.load_model()

        self.topic_ids = list(topic_ids)
        self.topic_texts = list(topic_texts)
        self._topic_id_to_idx = {tid: i for i, tid in enumerate(self.topic_ids)}

        # Normalize embeddings so dot product == cosine similarity
        self.topic_embeddings = self.model.encode(
            self.topic_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)

        # Invalidate similarity matrix if topics changed
        self.similarity_matrix = None

    def encode_opinions(self, opinion_ids: List[str], opinion_texts: List[str], batch_size: int = 64):
        if len(opinion_ids) != len(opinion_texts):
            raise ValueError("opinion_ids and opinion_texts must have the same length")

        self.load_model()

        self.opinion_ids = list(opinion_ids)
        self.opinion_texts = list(opinion_texts)
        self._opinion_id_to_idx = {oid: i for i, oid in enumerate(self.opinion_ids)}

        self.opinion_embeddings = self.model.encode(
            self.opinion_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)

        # Invalidate similarity matrix if opinions changed
        self.similarity_matrix = None

    # ----------------------------
    # Similarity matrix compute (once)
    # ----------------------------
    def compute_similarity_matrix(self):
        """
        Compute full matrix once:
          similarity_matrix[i, j] = cosine(opinion_i, topic_j)
        Shape: (n_opinions, n_topics)
        """
        if self.topic_embeddings is None:
            raise ValueError("Topics not encoded. Call encode_topics() first.")
        if self.opinion_embeddings is None:
            raise ValueError("Opinions not encoded. Call encode_opinions() first.")

        # (n_opinions, dim) @ (dim, n_topics) -> (n_opinions, n_topics)
        self.similarity_matrix = (self.opinion_embeddings @ self.topic_embeddings.T).astype(np.float32)

    # ----------------------------
    # Retrieval per topic (slice)
    # ----------------------------
    def top_opinions_for_topic(
        self,
        topic_id: str,
        top_k: int = 10,
        threshold: Optional[float] = None,
        relative_margin: Optional[float] = None,
    ) -> List[Dict]:
        """
        Return top-k opinions for a given topic by slicing the similarity matrix.

        threshold:
          - absolute cosine threshold (e.g., 0.45). If None uses settings.SIMILARITY_THRESHOLD if present.
        relative_margin:
          - optional adaptive filter: keep opinions with score >= (best_score - relative_margin)
            Example: relative_margin=0.05 keeps near-best matches and avoids global-threshold brittleness.

        Output items:
          { opinion_id, opinion_text, similarity }
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
        if topic_id not in self._topic_id_to_idx:
            raise KeyError(f"Unknown topic_id: {topic_id}")

        topic_idx = self._topic_id_to_idx[topic_id]
        scores = self.similarity_matrix[:, topic_idx]  # slice per topic (n_opinions,)

        # Rank opinions by score descending
        if top_k <= 0:
            top_k = len(scores)
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Determine thresholds
        abs_threshold = threshold
        if abs_threshold is None:
            abs_threshold = getattr(settings, "SIMILARITY_THRESHOLD", None)

        best_score = float(scores[top_idx[0]]) if len(top_idx) else -1.0
        rel_threshold = None
        if relative_margin is not None:
            rel_threshold = best_score - float(relative_margin)

        results: List[Dict] = []
        for oi in top_idx:
            s = float(scores[oi])

            if abs_threshold is not None and s < float(abs_threshold):
                continue
            if rel_threshold is not None and s < rel_threshold:
                continue

            results.append(
                {
                    "opinion_id": self.opinion_ids[oi],
                    "opinion_text": self.opinion_texts[oi],
                    "similarity": s,
                }
            )

        return results

    def match_all_topics(
        self,
        top_k: int = 10,
        threshold: Optional[float] = None,
        relative_margin: Optional[float] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Convenience: produce mapping topic_id -> top opinions list
        using the already computed similarity matrix.
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
        out: Dict[str, List[Dict]] = {}
        for tid in self.topic_ids:
            out[tid] = self.top_opinions_for_topic(
                tid, top_k=top_k, threshold=threshold, relative_margin=relative_margin
            )
        return out

    # ----------------------------
    # Save / load (optional)
    # ----------------------------
    def save(self, path: str | Path):
        """
        Save encoded embeddings + metadata. (Similarity matrix is optional but included if computed.)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "model_name": self.model_name,
            "topic_ids": self.topic_ids,
            "topic_texts": self.topic_texts,
            "topic_embeddings": self.topic_embeddings,
            "opinion_ids": self.opinion_ids,
            "opinion_texts": self.opinion_texts,
            "opinion_embeddings": self.opinion_embeddings,
            "similarity_matrix": self.similarity_matrix,
        }
        with open(path / "topic_matcher.pkl", "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "TopicMatcher":
        """
        Load embeddings + metadata from disk.
        """
        path = Path(path)
        with open(path / "topic_matcher.pkl", "rb") as f:
            data = pickle.load(f)

        obj = cls(model_name=data.get("model_name"))
        obj.topic_ids = data.get("topic_ids", [])
        obj.topic_texts = data.get("topic_texts", [])
        obj.topic_embeddings = data.get("topic_embeddings", None)

        obj.opinion_ids = data.get("opinion_ids", [])
        obj.opinion_texts = data.get("opinion_texts", [])
        obj.opinion_embeddings = data.get("opinion_embeddings", None)

        obj.similarity_matrix = data.get("similarity_matrix", None)

        obj._topic_id_to_idx = {tid: i for i, tid in enumerate(obj.topic_ids)}
        obj._opinion_id_to_idx = {oid: i for i, oid in enumerate(obj.opinion_ids)}
        return obj
