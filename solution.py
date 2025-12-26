#!/usr/bin/env python3
"""
SOLUTION - SOCIAL MEDIA OPINION ANALYSIS PIPELINE
=================================================
This is the main testable entry point for the DigitalPulse social media analysis system.

PIPELINE:
1. Match opinions to topics using sentence embeddings
2. Classify opinions (Claim, Counterclaim, Rebuttal, Evidence)
3. Generate conclusions using OpenAI

USAGE:
    from solution import SocialMediaAnalyzer

    # INITIALIZE
    analyzer = SocialMediaAnalyzer()
    analyzer.load_data()
    analyzer.initialize_models()

    # ANALYZE
    results = analyzer.analyze(
        topic_text="Mars face is a natural landform",
        opinion_texts=[
            "I think there is no life on Mars",
            "Some believe aliens created it"
        ]
    )

    # EVALUATE
    metrics = analyzer.evaluate()
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ADD PROJECT ROOT TO PATH
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings, OPINION_TYPES
from data import DataProcessor
from models import TopicMatcher, OpinionClassifier, ConclusionGenerator
from evaluation import Evaluator


class SocialMediaAnalyzer:
    """
    MAIN ANALYZER CLASS FOR SOCIAL MEDIA OPINION ANALYSIS

    This class provides a unified interface for:
    - Topic matching (sentence similarity)
    - Opinion classification (DistilBERT)
    - Conclusion generation (OpenAI)
    - Evaluation metrics
    """

    def __init__(self):
        """INITIALIZE ANALYZER"""
        self.processor = None
        self.topic_matcher = None
        self.classifier = None
        self.conclusion_generator = None
        self.evaluator = Evaluator()

        # DATA SPLITS
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(
        self,
        topics_path: str = None,
        opinions_path: str = None,
        conclusions_path: str = None
    ):
        """
        LOAD AND PREPARE DATA

        Args:
            topics_path: Path to topics.csv
            opinions_path: Path to opinions.csv
            conclusions_path: Path to conclusions.csv
        """
        # SET DEFAULT PATHS
        topics_path = topics_path or str(settings.DATA_DIR / "topics.csv")
        opinions_path = opinions_path or str(settings.DATA_DIR / "opinions.csv")
        conclusions_path = conclusions_path or str(settings.DATA_DIR / "conclusions.csv")

        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        self.processor = DataProcessor()
        self.processor.load_data(
            topics_path=topics_path,
            opinions_path=opinions_path,
            conclusions_path=conclusions_path
        )

        # ANALYZE DATA
        analysis = self.processor.analyze_data()
        print(f"\nData Summary:")
        print(f"  Topics: {analysis['topics']['count']}")
        print(f"  Opinions: {analysis['opinions']['count']}")
        print(f"  Conclusions: {analysis['conclusions']['count']}")
        print(f"\nOpinion Type Distribution:")
        for op_type, count in analysis['opinions']['type_distribution'].items():
            print(f"  {op_type}: {count}")

    def initialize_models(
        self,
        load_topic_matcher: bool = True,
        load_classifier: bool = True,
        load_conclusion_generator: bool = True,
        classifier_path: str = None,
        topic_matcher_path: str = None
    ):
        """
        INITIALIZE ML MODELS

        Args:
            load_topic_matcher: Whether to load topic matcher
            load_classifier: Whether to load classifier
            load_conclusion_generator: Whether to load conclusion generator
            classifier_path: Path to pre-trained classifier
            topic_matcher_path: Path to pre-trained topic matcher
        """
        print("\n" + "=" * 60)
        print("INITIALIZING MODELS")
        print("=" * 60)

        # LOAD TOPIC MATCHER
        if load_topic_matcher:
            print("\n[1/3] Loading Topic Matcher...")
            self.topic_matcher = TopicMatcher()

            if topic_matcher_path and Path(topic_matcher_path).exists():
                self.topic_matcher.load(Path(topic_matcher_path))
            else:
                self.topic_matcher.load_model()

                # ENCODE TOPICS IF DATA IS LOADED
                if self.processor and self.processor.topics_df is not None:
                    topics, _ = self.processor.prepare_topic_matching_data()
                    self.topic_matcher.encode_topics(
                        topic_ids=topics["topic_id"].tolist(),
                        topic_texts=topics["clean_text"].tolist()
                    )

        # LOAD CLASSIFIER
        if load_classifier:
            print("\n[2/3] Loading Opinion Classifier...")
            self.classifier = OpinionClassifier()

            if classifier_path and Path(classifier_path).exists():
                self.classifier.load(Path(classifier_path))
            else:
                self.classifier.load_model()

        # LOAD CONCLUSION GENERATOR
        if load_conclusion_generator:
            print("\n[3/3] Loading Conclusion Generator...")
            try:
                self.conclusion_generator = ConclusionGenerator()
                self.conclusion_generator.initialize()
            except Exception as e:
                print(f"  Warning: Could not initialize conclusion generator: {e}")
                self.conclusion_generator = None

        print("\n" + "=" * 60)
        print("MODELS INITIALIZED")
        print("=" * 60)

    def match_opinions_to_topics(
        self,
        opinion_texts: List[str],
        top_k: int = 1,
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        MATCH OPINIONS TO THEIR RELEVANT TOPICS

        Args:
            opinion_texts: List of opinion texts
            top_k: Number of top matches per opinion
            threshold: Minimum similarity threshold

        Returns:
            List of match results per opinion
        """
        if self.topic_matcher is None:
            raise ValueError("Topic matcher not initialized")

        return self.topic_matcher.match_opinions_batch(
            opinion_texts,
            top_k=top_k,
            threshold=threshold
        )

    def classify_opinions(self, opinion_texts: List[str]) -> List[Dict]:
        """
        CLASSIFY OPINIONS INTO TYPES

        Args:
            opinion_texts: List of opinion texts

        Returns:
            List of classification results
        """
        if self.classifier is None:
            raise ValueError("Classifier not initialized")

        results = []
        for text in opinion_texts:
            result = self.classifier.predict_single(text)
            results.append({
                "text": text,
                "type": result["predicted_type"],
                "confidence": max(result["probabilities"].values()),
                "probabilities": result["probabilities"]
            })

        return results

    def generate_conclusion(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]]
    ) -> str:
        """
        GENERATE CONCLUSION FROM TOPIC AND OPINIONS

        Args:
            topic_text: Main topic text
            opinions: List of dicts with 'text' and 'type' keys

        Returns:
            Generated conclusion text
        """
        if self.conclusion_generator is None:
            raise ValueError("Conclusion generator not initialized")

        return self.conclusion_generator.generate(
            topic_text=topic_text,
            opinions=opinions
        )

    def analyze(
        self,
        topic_text: str,
        opinion_texts: List[str],
        generate_conclusion: bool = True
    ) -> Dict:
        """
        FULL ANALYSIS PIPELINE

        Args:
            topic_text: The main topic/position
            opinion_texts: List of opinions to analyze
            generate_conclusion: Whether to generate conclusion

        Returns:
            Dict with classified opinions and conclusion
        """
        print("\n" + "=" * 60)
        print("RUNNING ANALYSIS")
        print("=" * 60)

        result = {
            "topic_text": topic_text,
            "classified_opinions": [],
            "conclusion": None
        }

        # STEP 1: CLASSIFY OPINIONS
        print("\n[Step 1] Classifying opinions...")
        classified = self.classify_opinions(opinion_texts)
        result["classified_opinions"] = classified

        # PRINT CLASSIFICATIONS
        for i, c in enumerate(classified):
            print(f"  Opinion {i+1}: {c['type']} (confidence: {c['confidence']:.2f})")

        # STEP 2: GENERATE CONCLUSION
        if generate_conclusion and self.conclusion_generator:
            print("\n[Step 2] Generating conclusion...")
            opinions_for_conclusion = [
                {"text": c["text"], "type": c["type"]}
                for c in classified
            ]
            result["conclusion"] = self.generate_conclusion(
                topic_text=topic_text,
                opinions=opinions_for_conclusion
            )
            print(f"  Conclusion: {result['conclusion'][:100]}...")

        return result

    def train_classifier(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        save_path: str = "trained_models/classifier"
    ) -> Dict:
        """
        TRAIN THE OPINION CLASSIFIER

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            save_path: Where to save the trained model

        Returns:
            Training history
        """
        if self.processor is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "=" * 60)
        print("TRAINING CLASSIFIER")
        print("=" * 60)

        # PREPARE DATA
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            self.processor.prepare_classification_data()

        # INITIALIZE CLASSIFIER
        if self.classifier is None:
            self.classifier = OpinionClassifier()
            self.classifier.load_model()

        # TRAIN
        history = self.classifier.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=epochs,
            batch_size=batch_size,
            save_path=Path(save_path)
        )

        # STORE TEST DATA FOR EVALUATION
        self.test_data = {
            "texts": test_texts,
            "labels": test_labels
        }

        return history

    def evaluate(self, sample_size: int = None) -> Dict:
        """
        EVALUATE ALL COMPONENTS

        Args:
            sample_size: Number of samples for evaluation (None = all)

        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)

        results = {}

        # EVALUATE CLASSIFIER
        if self.classifier and self.processor:
            print("\n[Classification Evaluation]")

            # GET TEST DATA
            if self.test_data is None:
                _, _, _, _, test_texts, test_labels = \
                    self.processor.prepare_classification_data()
            else:
                test_texts = self.test_data["texts"]
                test_labels = self.test_data["labels"]

            # SAMPLE IF NEEDED
            if sample_size:
                test_texts = test_texts[:sample_size]
                test_labels = test_labels[:sample_size]

            # PREDICT
            predictions = self.classifier.predict(test_texts)

            # EVALUATE
            results["classification"] = self.evaluator.evaluate_classification(
                predicted_labels=predictions,
                true_labels=test_labels
            )

            print(f"  Accuracy: {results['classification']['accuracy']:.4f}")
            print(f"  F1 (weighted): {results['classification']['f1']['weighted']:.4f}")
            print(f"  F1 (macro): {results['classification']['f1']['macro']:.4f}")

        # EVALUATE TOPIC MATCHING
        if self.topic_matcher and self.processor:
            print("\n[Topic Matching Evaluation]")

            splits = self.processor.prepare_topic_matching_splits()
            test_opinions = splits["test"]["opinions"]

            if sample_size:
                test_opinions = test_opinions.sample(
                    n=min(sample_size, len(test_opinions)),
                    random_state=42
                )

            # GET PREDICTIONS
            predictions = self.topic_matcher.match_opinions_batch(
                test_opinions["clean_text"].tolist(),
                top_k=5,
                threshold=0.0
            )

            top1_preds = [p[0]["topic_id"] if p else None for p in predictions]
            top5_preds = [[m["topic_id"] for m in p] for p in predictions]

            results["topic_matching"] = self.evaluator.evaluate_topic_matching(
                predicted_topic_ids=top1_preds,
                true_topic_ids=test_opinions["topic_id"].tolist(),
                top_k_predictions=top5_preds
            )

            print(f"  Top-1 Accuracy: {results['topic_matching']['accuracy']:.4f}")
            if 'accuracy_top_k' in results['topic_matching']:
                print(f"  Top-5 Accuracy: {results['topic_matching']['accuracy_top_k']:.4f}")

        return results

    def save_models(self, output_dir: str = "trained_models"):
        """
        SAVE ALL TRAINED MODELS

        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.classifier:
            self.classifier.save(output_path / "classifier")

        if self.topic_matcher:
            self.topic_matcher.save(output_path / "topic_matcher")

        print(f"Models saved to {output_path}")


def run_demo():
    """
    RUN A DEMO OF THE ANALYSIS PIPELINE
    """
    print("=" * 60)
    print("SOCIAL MEDIA ANALYSIS DEMO")
    print("=" * 60)

    # INITIALIZE
    analyzer = SocialMediaAnalyzer()
    analyzer.load_data()
    analyzer.initialize_models(
        load_topic_matcher=True,
        load_classifier=True,
        load_conclusion_generator=True
    )

    # SAMPLE ANALYSIS
    topic = "I think that the face on Mars is a natural landform because I dont think there is any life on Mars."

    opinions = [
        "I think that the face is a natural landform because there is no life on Mars that we have discovered yet",
        "If life was on Mars, we would know by now. Nobody lives on Mars to create the figure.",
        "People thought that the face was formed by aliens because they believed there was life on Mars.",
        "Though some say that life on Mars does exist, I think that there is no life on Mars."
    ]

    # RUN ANALYSIS
    results = analyzer.analyze(
        topic_text=topic,
        opinion_texts=opinions,
        generate_conclusion=True
    )

    # PRINT RESULTS
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTopic: {results['topic_text'][:100]}...")
    print("\nClassified Opinions:")
    for i, op in enumerate(results['classified_opinions']):
        print(f"  {i+1}. [{op['type']}] {op['text'][:60]}...")

    if results['conclusion']:
        print(f"\nGenerated Conclusion:\n  {results['conclusion']}")

    # EVALUATE
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    metrics = analyzer.evaluate(sample_size=500)

    return results, metrics


if __name__ == "__main__":
    run_demo()
