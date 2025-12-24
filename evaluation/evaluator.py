"""
EVALUATION MODULE
Comprehensive evaluation for all ML components using F1 and other metrics.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, OPINION_TYPES, ID_TO_OPINION_TYPE


class Evaluator:
    """
    COMPREHENSIVE EVALUATOR FOR ALL PIPELINE COMPONENTS
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_topic_matching(
        self,
        predicted_topic_ids: List[str],
        true_topic_ids: List[str],
        top_k_predictions: Optional[List[List[str]]] = None
    ) -> Dict:
        """
        EVALUATE TOPIC MATCHING USING F1 AND ACCURACY

        Args:
            predicted_topic_ids: Top-1 predicted topic IDs
            true_topic_ids: Ground truth topic IDs
            top_k_predictions: Optional list of top-k predictions for top-k accuracy

        Returns:
            Dict with evaluation metrics
        """
        # BASIC ACCURACY (EXACT MATCH)
        accuracy = accuracy_score(true_topic_ids, predicted_topic_ids)

        # FOR MULTI-CLASS F1, WE TREAT EACH UNIQUE TOPIC AS A CLASS
        # USE MICRO-AVERAGING SINCE WE HAVE MANY CLASSES
        f1_micro = f1_score(true_topic_ids, predicted_topic_ids, average="micro")
        f1_macro = f1_score(true_topic_ids, predicted_topic_ids, average="macro")
        f1_weighted = f1_score(true_topic_ids, predicted_topic_ids, average="weighted")

        precision_micro = precision_score(true_topic_ids, predicted_topic_ids, average="micro")
        recall_micro = recall_score(true_topic_ids, predicted_topic_ids, average="micro")

        results = {
            "accuracy": float(accuracy),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_micro": float(precision_micro),
            "recall_micro": float(recall_micro),
            "num_samples": len(true_topic_ids),
            "num_unique_topics": len(set(true_topic_ids))
        }

        # TOP-K ACCURACY IF PROVIDED
        if top_k_predictions:
            correct_at_k = sum(
                1 for true_id, preds in zip(true_topic_ids, top_k_predictions)
                if true_id in preds
            )
            results["accuracy_top_k"] = correct_at_k / len(true_topic_ids)
            results["top_k"] = len(top_k_predictions[0]) if top_k_predictions else 0

        return results

    def evaluate_classification(
        self,
        predicted_labels: List[int],
        true_labels: List[int],
        label_names: List[str] = None
    ) -> Dict:
        """
        EVALUATE OPINION CLASSIFICATION USING F1 AND OTHER METRICS

        Args:
            predicted_labels: Predicted class indices
            true_labels: Ground truth class indices
            label_names: Optional class names for reporting

        Returns:
            Dict with comprehensive evaluation metrics
        """
        label_names = label_names or OPINION_TYPES

        # CORE METRICS
        accuracy = accuracy_score(true_labels, predicted_labels)

        f1_micro = f1_score(true_labels, predicted_labels, average="micro")
        f1_macro = f1_score(true_labels, predicted_labels, average="macro")
        f1_weighted = f1_score(true_labels, predicted_labels, average="weighted")

        precision_micro = precision_score(true_labels, predicted_labels, average="micro")
        precision_macro = precision_score(true_labels, predicted_labels, average="macro")
        precision_weighted = precision_score(true_labels, predicted_labels, average="weighted")

        recall_micro = recall_score(true_labels, predicted_labels, average="micro")
        recall_macro = recall_score(true_labels, predicted_labels, average="macro")
        recall_weighted = recall_score(true_labels, predicted_labels, average="weighted")

        # PER-CLASS F1
        f1_per_class = f1_score(true_labels, predicted_labels, average=None)
        precision_per_class = precision_score(true_labels, predicted_labels, average=None)
        recall_per_class = recall_score(true_labels, predicted_labels, average=None)

        # CONFUSION MATRIX
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # CLASSIFICATION REPORT
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=label_names,
            output_dict=True
        )

        results = {
            "accuracy": float(accuracy),
            "f1": {
                "micro": float(f1_micro),
                "macro": float(f1_macro),
                "weighted": float(f1_weighted),
                "per_class": {
                    label_names[i]: float(f1_per_class[i])
                    for i in range(len(label_names))
                }
            },
            "precision": {
                "micro": float(precision_micro),
                "macro": float(precision_macro),
                "weighted": float(precision_weighted),
                "per_class": {
                    label_names[i]: float(precision_per_class[i])
                    for i in range(len(label_names))
                }
            },
            "recall": {
                "micro": float(recall_micro),
                "macro": float(recall_macro),
                "weighted": float(recall_weighted),
                "per_class": {
                    label_names[i]: float(recall_per_class[i])
                    for i in range(len(label_names))
                }
            },
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report,
            "num_samples": len(true_labels),
            "class_distribution": {
                label_names[i]: int(sum(1 for l in true_labels if l == i))
                for i in range(len(label_names))
            }
        }

        return results

    def evaluate_conclusions(
        self,
        generated: List[str],
        references: List[str],
        use_bertscore: bool = False
    ) -> Dict:
        """
        EVALUATE CONCLUSION GENERATION USING ROUGE SCORES

        Args:
            generated: Generated conclusions
            references: Reference conclusions
            use_bertscore: Whether to compute BERTScore (slower)

        Returns:
            Dict with evaluation metrics
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        scores = {
            "rouge1": {"precision": [], "recall": [], "fmeasure": []},
            "rouge2": {"precision": [], "recall": [], "fmeasure": []},
            "rougeL": {"precision": [], "recall": [], "fmeasure": []}
        }

        for gen, ref in zip(generated, references):
            result = scorer.score(ref, gen)
            for metric in ["rouge1", "rouge2", "rougeL"]:
                scores[metric]["precision"].append(result[metric].precision)
                scores[metric]["recall"].append(result[metric].recall)
                scores[metric]["fmeasure"].append(result[metric].fmeasure)

        results = {
            "rouge1": {
                "precision": float(np.mean(scores["rouge1"]["precision"])),
                "recall": float(np.mean(scores["rouge1"]["recall"])),
                "f1": float(np.mean(scores["rouge1"]["fmeasure"]))
            },
            "rouge2": {
                "precision": float(np.mean(scores["rouge2"]["precision"])),
                "recall": float(np.mean(scores["rouge2"]["recall"])),
                "f1": float(np.mean(scores["rouge2"]["fmeasure"]))
            },
            "rougeL": {
                "precision": float(np.mean(scores["rougeL"]["precision"])),
                "recall": float(np.mean(scores["rougeL"]["recall"])),
                "f1": float(np.mean(scores["rougeL"]["fmeasure"]))
            },
            "num_samples": len(generated),
            "avg_generated_length": float(np.mean([len(g.split()) for g in generated])),
            "avg_reference_length": float(np.mean([len(r.split()) for r in references]))
        }

        if use_bertscore:
            try:
                from bert_score import score as bert_score_fn
                P, R, F1 = bert_score_fn(generated, references, lang="en", verbose=False)
                results["bertscore"] = {
                    "precision": float(P.mean()),
                    "recall": float(R.mean()),
                    "f1": float(F1.mean())
                }
            except ImportError:
                print("BERTScore not installed. Skipping.")

        return results

    def run_full_evaluation(
        self,
        topic_matcher,
        classifier,
        conclusion_generator,
        test_data: Dict,
        sample_size: int = None
    ) -> Dict:
        """
        RUN EVALUATION ON ALL COMPONENTS

        Args:
            topic_matcher: Initialized TopicMatcher
            classifier: Initialized OpinionClassifier
            conclusion_generator: Initialized ConclusionGenerator
            test_data: Dict with test splits
            sample_size: Optional sample size for faster evaluation

        Returns:
            Comprehensive evaluation results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }

        # EVALUATE TOPIC MATCHING
        if topic_matcher and "opinions" in test_data:
            print("Evaluating Topic Matching...")
            test_opinions = test_data["opinions"]

            if sample_size:
                test_opinions = test_opinions.sample(n=min(sample_size, len(test_opinions)), random_state=42)

            # GET PREDICTIONS WITH TOP-5
            predictions = topic_matcher.match_opinions_batch(
                test_opinions["clean_text"].tolist(),
                top_k=5,
                threshold=0.0
            )

            top1_preds = [p[0]["topic_id"] if p else None for p in predictions]
            top5_preds = [[m["topic_id"] for m in p] for p in predictions]

            results["components"]["topic_matching"] = self.evaluate_topic_matching(
                predicted_topic_ids=top1_preds,
                true_topic_ids=test_opinions["topic_id"].tolist(),
                top_k_predictions=top5_preds
            )

        # EVALUATE CLASSIFICATION
        if classifier:
            print("Evaluating Classification...")
            from data import DataProcessor
            processor = DataProcessor()
            processor.load_data()

            _, _, _, _, test_texts, test_labels = processor.prepare_classification_data()

            if sample_size:
                test_texts = test_texts[:sample_size]
                test_labels = test_labels[:sample_size]

            predictions = classifier.predict(test_texts)

            results["components"]["classification"] = self.evaluate_classification(
                predicted_labels=predictions,
                true_labels=test_labels
            )

        # EVALUATE CONCLUSION GENERATION (IF OPENAI AVAILABLE)
        if conclusion_generator:
            print("Evaluating Conclusion Generation...")
            # THIS REQUIRES GENERATING CONCLUSIONS WHICH IS EXPENSIVE
            # SKIP BY DEFAULT UNLESS SPECIFICALLY REQUESTED
            results["components"]["conclusion_generation"] = {
                "note": "Conclusion generation evaluation requires OpenAI API calls. Run separately."
            }

        return results

    def save_results(self, results: Dict, name: str = "evaluation"):
        """SAVE EVALUATION RESULTS TO JSON FILE"""
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath

    def print_results(self, results: Dict):
        """PRETTY PRINT EVALUATION RESULTS"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        for component, metrics in results.get("components", {}).items():
            print(f"\n{component.upper().replace('_', ' ')}")
            print("-" * 50)

            if component == "classification":
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 (weighted): {metrics['f1']['weighted']:.4f}")
                print(f"  F1 (macro): {metrics['f1']['macro']:.4f}")
                print("\n  Per-class F1:")
                for cls, f1 in metrics['f1']['per_class'].items():
                    print(f"    {cls}: {f1:.4f}")

            elif component == "topic_matching":
                print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")
                if 'accuracy_top_k' in metrics:
                    print(f"  Accuracy (Top-{metrics['top_k']}): {metrics['accuracy_top_k']:.4f}")
                print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

            elif component == "conclusion_generation":
                if "rouge1" in metrics:
                    print(f"  ROUGE-1 F1: {metrics['rouge1']['f1']:.4f}")
                    print(f"  ROUGE-2 F1: {metrics['rouge2']['f1']:.4f}")
                    print(f"  ROUGE-L F1: {metrics['rougeL']['f1']:.4f}")
                else:
                    print(f"  {metrics.get('note', 'No results')}")


def run_evaluation_pipeline():
    """RUN THE FULL EVALUATION PIPELINE"""
    from data import DataProcessor
    from models import TopicMatcher, OpinionClassifier

    print("Loading data...")
    processor = DataProcessor()
    processor.load_data()

    # PREPARE TEST DATA
    splits = processor.prepare_topic_matching_splits()
    test_data = splits["test"]

    # INITIALIZE EVALUATOR
    evaluator = Evaluator()

    # NOTE: IN PRODUCTION, LOAD TRAINED MODELS
    # FOR NOW, WE'LL JUST DEMONSTRATE THE CLASSIFICATION EVALUATION

    print("\n" + "=" * 70)
    print("Running classification evaluation with pre-trained DistilBERT...")
    print("=" * 70)

    classifier = OpinionClassifier()
    classifier.load_model()

    # GET TEST DATA
    _, _, _, _, test_texts, test_labels = processor.prepare_classification_data()

    # EVALUATE ON SAMPLE
    sample_size = 500
    predictions = classifier.predict(test_texts[:sample_size])

    results = evaluator.evaluate_classification(
        predicted_labels=predictions,
        true_labels=test_labels[:sample_size]
    )

    full_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "classification": results
        }
    }

    evaluator.print_results(full_results)
    evaluator.save_results(full_results)

    return full_results


if __name__ == "__main__":
    run_evaluation_pipeline()
