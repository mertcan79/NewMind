"""
DATA PREPROCESSING AND ANALYSIS MODULE
Handles loading, cleaning, and splitting the dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from collections import Counter
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, OPINION_TYPES


class DataProcessor:
    """
    HANDLES ALL DATA LOADING, PREPROCESSING, AND SPLITTING OPERATIONS
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.topics_df = None
        self.opinions_df = None
        self.conclusions_df = None

    def load_data(
        self,
        topics_path: str = None,
        opinions_path: str = None,
        conclusions_path: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """LOAD ALL THREE CSV FILES"""

        topics_path = topics_path or self.data_dir / "topics.csv"
        opinions_path = opinions_path or self.data_dir / "opinions.csv"
        conclusions_path = conclusions_path or self.data_dir / "conclusions.csv"

        self.topics_df = pd.read_csv(topics_path)
        self.opinions_df = pd.read_csv(opinions_path)
        self.conclusions_df = pd.read_csv(conclusions_path)

        print(f"Loaded {len(self.topics_df)} topics")
        print(f"Loaded {len(self.opinions_df)} opinions")
        print(f"Loaded {len(self.conclusions_df)} conclusions")

        return self.topics_df, self.opinions_df, self.conclusions_df

    def analyze_data(self) -> Dict:
        """PERFORM EXPLORATORY DATA ANALYSIS"""

        if self.topics_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        analysis = {
            "topics": {
                "count": len(self.topics_df),
                "unique_topic_ids": self.topics_df["topic_id"].nunique(),
                "avg_text_length": self.topics_df["text"].str.len().mean(),
                "effectiveness_distribution": self.topics_df["effectiveness"].value_counts().to_dict()
            },
            "opinions": {
                "count": len(self.opinions_df),
                "unique_topic_ids": self.opinions_df["topic_id"].nunique(),
                "avg_text_length": self.opinions_df["text"].str.len().mean(),
                "type_distribution": self.opinions_df["type"].value_counts().to_dict(),
                "effectiveness_distribution": self.opinions_df["effectiveness"].value_counts().to_dict(),
                "avg_opinions_per_topic": len(self.opinions_df) / self.opinions_df["topic_id"].nunique()
            },
            "conclusions": {
                "count": len(self.conclusions_df),
                "unique_topic_ids": self.conclusions_df["topic_id"].nunique(),
                "avg_text_length": self.conclusions_df["text"].str.len().mean(),
                "effectiveness_distribution": self.conclusions_df["effectiveness"].value_counts().to_dict()
            }
        }

        return analysis

    def preprocess_text(self, text: str) -> str:
        """CLEAN AND PREPROCESS TEXT"""
        if pd.isna(text):
            return ""

        # BASIC CLEANING
        text = str(text).strip()

        # REMOVE EXTRA WHITESPACE
        text = " ".join(text.split())

        return text

    def prepare_topic_matching_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        PREPARE DATA FOR TOPIC-OPINION MATCHING TASK
        Returns topics and opinions with preprocessed text.
        """

        # PREPROCESS TEXTS
        topics = self.topics_df.copy()
        topics["clean_text"] = topics["text"].apply(self.preprocess_text)

        opinions = self.opinions_df.copy()
        opinions["clean_text"] = opinions["text"].apply(self.preprocess_text)

        # FILTER OUT OPINIONS THAT ARE NOT IN VALID TYPES (EXCLUDE POSITION TYPE)
        valid_types = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
        opinions = opinions[opinions["type"].isin(valid_types)]

        return topics, opinions

    def prepare_classification_data(self) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """
        PREPARE DATA FOR OPINION TYPE CLASSIFICATION
        Returns train, val, test splits with texts and labels.
        """

        # FILTER VALID OPINION TYPES
        valid_types = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
        opinions = self.opinions_df[self.opinions_df["type"].isin(valid_types)].copy()

        # PREPROCESS TEXTS
        opinions["clean_text"] = opinions["text"].apply(self.preprocess_text)

        # CONVERT LABELS TO INTEGERS
        from config.settings import OPINION_TYPE_TO_ID
        opinions["label"] = opinions["type"].map(OPINION_TYPE_TO_ID)

        # SPLIT BY TOPIC_ID TO AVOID DATA LEAKAGE
        unique_topics = opinions["topic_id"].unique()

        train_topics, temp_topics = train_test_split(
            unique_topics,
            test_size=(settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        val_topics, test_topics = train_test_split(
            temp_topics,
            test_size=settings.TEST_RATIO / (settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        # CREATE SPLITS
        train_df = opinions[opinions["topic_id"].isin(train_topics)]
        val_df = opinions[opinions["topic_id"].isin(val_topics)]
        test_df = opinions[opinions["topic_id"].isin(test_topics)]

        print(f"Classification data split:")
        print(f"  Train: {len(train_df)} samples ({len(train_topics)} topics)")
        print(f"  Val: {len(val_df)} samples ({len(val_topics)} topics)")
        print(f"  Test: {len(test_df)} samples ({len(test_topics)} topics)")

        # PRINT CLASS DISTRIBUTION
        print(f"\nClass distribution in training set:")
        for label, count in train_df["type"].value_counts().items():
            print(f"  {label}: {count}")

        return (
            train_df["clean_text"].tolist(),
            train_df["label"].tolist(),
            val_df["clean_text"].tolist(),
            val_df["label"].tolist(),
            test_df["clean_text"].tolist(),
            test_df["label"].tolist()
        )

    def prepare_topic_matching_splits(self) -> Dict:
        """
        PREPARE TRAIN/VAL/TEST SPLITS FOR TOPIC MATCHING EVALUATION
        Split by topic_id to ensure proper evaluation.
        """

        topics, opinions = self.prepare_topic_matching_data()

        unique_topics = topics["topic_id"].unique()

        train_topics, temp_topics = train_test_split(
            unique_topics,
            test_size=(settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        val_topics, test_topics = train_test_split(
            temp_topics,
            test_size=settings.TEST_RATIO / (settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        return {
            "train": {
                "topics": topics[topics["topic_id"].isin(train_topics)],
                "opinions": opinions[opinions["topic_id"].isin(train_topics)]
            },
            "val": {
                "topics": topics[topics["topic_id"].isin(val_topics)],
                "opinions": opinions[opinions["topic_id"].isin(val_topics)]
            },
            "test": {
                "topics": topics[topics["topic_id"].isin(test_topics)],
                "opinions": opinions[opinions["topic_id"].isin(test_topics)]
            }
        }

    def prepare_conclusion_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        PREPARE DATA FOR CONCLUSION GENERATION TASK
        Groups topics with their opinions and target conclusions.
        """

        # GET ALL DATA
        topics = self.topics_df.copy()
        opinions = self.opinions_df.copy()
        conclusions = self.conclusions_df.copy()

        # PREPROCESS
        topics["clean_text"] = topics["text"].apply(self.preprocess_text)
        opinions["clean_text"] = opinions["text"].apply(self.preprocess_text)
        conclusions["clean_text"] = conclusions["text"].apply(self.preprocess_text)

        # FILTER VALID OPINION TYPES
        valid_types = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
        opinions = opinions[opinions["type"].isin(valid_types)]

        # GET TOPICS THAT HAVE CONCLUSIONS
        valid_topic_ids = conclusions["topic_id"].unique()

        # SPLIT BY TOPIC_ID
        train_topics, temp_topics = train_test_split(
            valid_topic_ids,
            test_size=(settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        val_topics, test_topics = train_test_split(
            temp_topics,
            test_size=settings.TEST_RATIO / (settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=settings.RANDOM_SEED
        )

        def create_split(topic_ids):
            split_topics = topics[topics["topic_id"].isin(topic_ids)]
            split_opinions = opinions[opinions["topic_id"].isin(topic_ids)]
            split_conclusions = conclusions[conclusions["topic_id"].isin(topic_ids)]
            return split_topics, split_opinions, split_conclusions

        return {
            "train": create_split(train_topics),
            "val": create_split(val_topics),
            "test": create_split(test_topics)
        }


def print_analysis(analysis: Dict):
    """PRETTY PRINT THE ANALYSIS RESULTS"""

    print("\n" + "=" * 60)
    print("DATA ANALYSIS REPORT")
    print("=" * 60)

    for dataset_name, stats in analysis.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # RUN ANALYSIS ON THE DATA
    processor = DataProcessor()

    # LOAD FROM DATA DIRECTORY
    processor.load_data()

    # ANALYZE
    analysis = processor.analyze_data()
    print_analysis(analysis)

    # PREPARE CLASSIFICATION DATA
    print("\n" + "=" * 60)
    print("PREPARING CLASSIFICATION DATA")
    print("=" * 60)
    processor.prepare_classification_data()
