"""
COMPREHENSIVE EVALUATION SCRIPT

Evaluates all three components of the pipeline:
1. Topic-Opinion Matching (Recall@k, Precision@k, MRR)
2. Opinion Classification (macro-F1, per-class F1, confusion matrix)
3. Conclusion Generation (ROUGE-L, optional BERTScore)

IMPORTANT: topic_id is used ONLY for evaluation/validation, NOT as a feature during matching.

Outputs:
- evaluation_results/matching_metrics.json
- evaluation_results/classifier_metrics.json
- evaluation_results/conclusion_metrics.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# ============================================================================
# MATCHING EVALUATION
# ============================================================================

def evaluate_matching(
    matching_results: dict,
    opinions_df: pd.DataFrame,
    top_k: int = 10
) -> dict:
    """
    Evaluate topic-opinion matching using ground truth from opinions.csv.

    For each topic, the relevant opinions are those where topic_id matches.
    We compare the retrieved top-k against this ground truth.

    Metrics:
    - Recall@k: proportion of relevant opinions retrieved in top-k
    - Precision@k: proportion of retrieved opinions that are relevant
    - MRR: Mean Reciprocal Rank of first relevant opinion

    Args:
        matching_results: Dict mapping topic_id -> list of matched opinions
        opinions_df: DataFrame with ground truth opinions (has topic_id column)
        top_k: Number of top opinions retrieved per topic

    Returns:
        Dict with matching metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING TOPIC-OPINION MATCHING")
    print("=" * 60)

    # Build ground truth: topic_id -> set of relevant opinion_ids
    ground_truth = defaultdict(set)
    for _, row in opinions_df.iterrows():
        topic_id = str(row["topic_id"])
        opinion_id = str(row["id"])
        ground_truth[topic_id].add(opinion_id)

    print(f"Ground truth: {len(ground_truth)} topics with relevant opinions")

    # Compute metrics per topic
    recall_scores = []
    precision_scores = []
    reciprocal_ranks = []

    for topic_id, retrieved_opinions in matching_results.items():
        if topic_id not in ground_truth:
            continue

        relevant_ids = ground_truth[topic_id]
        retrieved_ids = [op["opinion_id"] for op in retrieved_opinions[:top_k]]

        # Recall@k: what fraction of relevant opinions did we retrieve?
        num_relevant_retrieved = len(set(retrieved_ids) & relevant_ids)
        recall = num_relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
        recall_scores.append(recall)

        # Precision@k: what fraction of retrieved opinions are relevant?
        precision = num_relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
        precision_scores.append(precision)

        # MRR: reciprocal rank of first relevant opinion
        first_relevant_rank = None
        for rank, opinion_id in enumerate(retrieved_ids, 1):
            if opinion_id in relevant_ids:
                first_relevant_rank = rank
                break

        if first_relevant_rank:
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    # Aggregate metrics
    metrics = {
        f"recall_at_{top_k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
        f"precision_at_{top_k}": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "num_topics_evaluated": len(recall_scores),
    }

    print(f"\nMatching Metrics (top-{top_k}):")
    print(f"  Recall@{top_k}: {metrics[f'recall_at_{top_k}']:.4f}")
    print(f"  Precision@{top_k}: {metrics[f'precision_at_{top_k}']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")
    print(f"  Topics evaluated: {metrics['num_topics_evaluated']}")

    return metrics


# ============================================================================
# CLASSIFICATION EVALUATION
# ============================================================================

def evaluate_classification(
    labeled_results: dict,
    opinions_df: pd.DataFrame
) -> dict:
    """
    Evaluate opinion classification using ground truth from opinions.csv.

    We evaluate on the retrieved set (opinions that were matched).

    Metrics:
    - Macro-F1: average F1 across all classes
    - Per-class F1, Precision, Recall
    - Confusion matrix

    Args:
        labeled_results: Dict with topic_id -> {topic_text, opinions with predictions}
        opinions_df: DataFrame with ground truth opinions (has type column)

    Returns:
        Dict with classification metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING OPINION CLASSIFICATION")
    print("=" * 60)

    # Build ground truth mapping: opinion_id -> true_type
    ground_truth_types = {}
    for _, row in opinions_df.iterrows():
        opinion_id = str(row["id"])
        true_type = str(row["type"])
        ground_truth_types[opinion_id] = true_type

    # Collect predictions
    y_true = []
    y_pred = []

    for topic_id, data in labeled_results.items():
        for opinion in data["opinions"]:
            opinion_id = opinion["opinion_id"]

            # Skip if we don't have ground truth for this opinion
            if opinion_id not in ground_truth_types:
                continue

            true_type = ground_truth_types[opinion_id]
            pred_type = opinion["predicted_type"]

            y_true.append(true_type)
            y_pred.append(pred_type)

    if not y_true:
        print("No opinions with ground truth found for evaluation!")
        return {}

    # Compute metrics
    labels = ["Claim", "Evidence", "Counterclaim", "Rebuttal"]

    # Overall metrics
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Per-class metrics
    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_samples": len(y_true),
        "per_class": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1_score": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"])
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist()
        }
    }

    print(f"\nClassification Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Samples evaluated: {metrics['num_samples']}")
    print(f"\nPer-class F1 scores:")
    for label in labels:
        f1 = metrics['per_class'][label]['f1_score']
        support = metrics['per_class'][label]['support']
        print(f"    {label:15s}: {f1:.4f} (support: {support})")

    return metrics


# ============================================================================
# CONCLUSION EVALUATION
# ============================================================================

def evaluate_conclusions(
    generated_df: pd.DataFrame,
    conclusions_df: pd.DataFrame,
    use_bertscore: bool = True
) -> dict:
    """
    Evaluate generated conclusions against reference conclusions.

    Metrics:
    - ROUGE-L F1 (required)
    - BERTScore (optional, more semantic)

    Args:
        generated_df: DataFrame with columns [topic_id, generated_conclusion]
        conclusions_df: DataFrame with columns [topic_id, text] (reference)
        use_bertscore: Whether to compute BERTScore

    Returns:
        Dict with conclusion metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING CONCLUSION GENERATION")
    print("=" * 60)

    # Build reference mapping: topic_id -> reference_conclusion
    references_map = {}
    for _, row in conclusions_df.iterrows():
        topic_id = str(row["topic_id"])
        reference = str(row["text"])
        references_map[topic_id] = reference

    # Collect matched pairs
    generated_texts = []
    reference_texts = []

    for _, row in generated_df.iterrows():
        topic_id = str(row["topic_id"])
        generated = str(row["generated_conclusion"])

        # Skip if no reference or empty generation
        if topic_id not in references_map or not generated or generated == "":
            continue

        generated_texts.append(generated)
        reference_texts.append(references_map[topic_id])

    if not generated_texts:
        print("No valid generated-reference pairs found for evaluation!")
        return {}

    print(f"Evaluating {len(generated_texts)} generated conclusions...")

    # Compute ROUGE scores
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for gen, ref in zip(generated_texts, reference_texts):
        scores = scorer.score(ref, gen)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    metrics = {
        "rouge1_f1": float(np.mean(rouge_scores["rouge1"])),
        "rouge2_f1": float(np.mean(rouge_scores["rouge2"])),
        "rougeL_f1": float(np.mean(rouge_scores["rougeL"])),
        "num_samples": len(generated_texts)
    }

    print(f"\nROUGE Metrics:")
    print(f"  ROUGE-1 F1: {metrics['rouge1_f1']:.4f}")
    print(f"  ROUGE-2 F1: {metrics['rouge2_f1']:.4f}")
    print(f"  ROUGE-L F1: {metrics['rougeL_f1']:.4f}")

    # Optionally compute BERTScore
    if use_bertscore:
        try:
            from bert_score import score as bert_score
            print("\nComputing BERTScore (this may take a while)...")
            P, R, F1 = bert_score(generated_texts, reference_texts, lang="en", verbose=False)
            metrics["bertscore_precision"] = float(P.mean().item())
            metrics["bertscore_recall"] = float(R.mean().item())
            metrics["bertscore_f1"] = float(F1.mean().item())

            print(f"\nBERTScore Metrics:")
            print(f"  Precision: {metrics['bertscore_precision']:.4f}")
            print(f"  Recall: {metrics['bertscore_recall']:.4f}")
            print(f"  F1: {metrics['bertscore_f1']:.4f}")
        except ImportError:
            print("\nBERTScore not available. Install with: pip install bert-score")
        except Exception as e:
            print(f"\nError computing BERTScore: {e}")

    print(f"\nSamples evaluated: {metrics['num_samples']}")

    return metrics


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def main(
    top_k: int = 10,
    use_bertscore: bool = False,
    evaluate_matching_only: bool = False,
    evaluate_classification_only: bool = False,
    evaluate_conclusions_only: bool = False,
):
    """
    Run comprehensive evaluation on all pipeline outputs.

    Args:
        top_k: Top-k value used in matching (for evaluation)
        use_bertscore: Whether to compute BERTScore for conclusions
        evaluate_matching_only: Only evaluate matching (skip others)
        evaluate_classification_only: Only evaluate classification (skip others)
        evaluate_conclusions_only: Only evaluate conclusions (skip others)
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    outputs_dir = base_dir / "outputs"
    eval_results_dir = base_dir / "evaluation_results"

    # Create evaluation results directory
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPREHENSIVE PIPELINE EVALUATION")
    print("=" * 60)
    print(f"Top-k for matching: {top_k}")
    print(f"BERTScore enabled: {use_bertscore}")
    print("=" * 60)

    # Load ground truth data
    print("\nLoading ground truth data...")
    topics_df = pd.read_csv(data_dir / "topics.csv")
    opinions_df = pd.read_csv(data_dir / "opinions.csv")
    conclusions_df = pd.read_csv(data_dir / "conclusions.csv")

    print(f"  Loaded {len(topics_df)} topics")
    print(f"  Loaded {len(opinions_df)} opinions")
    print(f"  Loaded {len(conclusions_df)} reference conclusions")

    # ========================================================================
    # 1. EVALUATE MATCHING
    # ========================================================================

    if not evaluate_classification_only and not evaluate_conclusions_only:
        matching_file = outputs_dir / "topic_to_opinions.json"

        if matching_file.exists():
            print(f"\nLoading matching results from {matching_file}...")
            with open(matching_file, "r") as f:
                matching_results = json.load(f)

            matching_metrics = evaluate_matching(matching_results, opinions_df, top_k)

            # Save matching metrics
            output_path = eval_results_dir / "matching_metrics.json"
            with open(output_path, "w") as f:
                json.dump(matching_metrics, f, indent=2)
            print(f"\nMatching metrics saved to: {output_path}")
        else:
            print(f"\nWarning: Matching results not found at {matching_file}")
            print("Run pipeline/run_matching.py first.")

    # ========================================================================
    # 2. EVALUATE CLASSIFICATION
    # ========================================================================

    if not evaluate_matching_only and not evaluate_conclusions_only:
        classification_file = outputs_dir / "topic_to_opinions_labeled.json"

        if classification_file.exists():
            print(f"\nLoading classification results from {classification_file}...")
            with open(classification_file, "r") as f:
                labeled_results = json.load(f)

            classification_metrics = evaluate_classification(labeled_results, opinions_df)

            # Save classification metrics
            output_path = eval_results_dir / "classifier_metrics.json"
            with open(output_path, "w") as f:
                json.dump(classification_metrics, f, indent=2)
            print(f"\nClassification metrics saved to: {output_path}")
        else:
            print(f"\nWarning: Classification results not found at {classification_file}")
            print("Run pipeline/run_classification.py first.")

    # ========================================================================
    # 3. EVALUATE CONCLUSIONS
    # ========================================================================

    if not evaluate_matching_only and not evaluate_classification_only:
        conclusions_file = outputs_dir / "conclusions_generated.csv"

        if conclusions_file.exists():
            print(f"\nLoading generated conclusions from {conclusions_file}...")
            generated_df = pd.read_csv(conclusions_file)

            conclusion_metrics = evaluate_conclusions(
                generated_df, conclusions_df, use_bertscore
            )

            # Save conclusion metrics
            output_path = eval_results_dir / "conclusion_metrics.json"
            with open(output_path, "w") as f:
                json.dump(conclusion_metrics, f, indent=2)
            print(f"\nConclusion metrics saved to: {output_path}")
        else:
            print(f"\nWarning: Generated conclusions not found at {conclusions_file}")
            print("Run pipeline/run_conclusions.py first.")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {eval_results_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all pipeline components"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k value used in matching (default: 10)",
    )
    parser.add_argument(
        "--use_bertscore",
        action="store_true",
        help="Compute BERTScore for conclusion evaluation (slower)",
    )
    parser.add_argument(
        "--matching_only",
        action="store_true",
        help="Only evaluate matching (skip classification and conclusions)",
    )
    parser.add_argument(
        "--classification_only",
        action="store_true",
        help="Only evaluate classification (skip matching and conclusions)",
    )
    parser.add_argument(
        "--conclusions_only",
        action="store_true",
        help="Only evaluate conclusions (skip matching and classification)",
    )

    args = parser.parse_args()

    main(
        top_k=args.top_k,
        use_bertscore=args.use_bertscore,
        evaluate_matching_only=args.matching_only,
        evaluate_classification_only=args.classification_only,
        evaluate_conclusions_only=args.conclusions_only,
    )
