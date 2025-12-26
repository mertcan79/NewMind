"""
OPINION CLASSIFICATION PIPELINE

This script takes matched topic-opinion pairs and classifies each opinion
into one of: Claim / Evidence / Counterclaim / Rebuttal

Input: outputs/topic_to_opinions.json
Output: outputs/topic_to_opinions_labeled.json

Output format:
{
  "<topic_id>": {
    "topic_text": "...",
    "opinions": [
      {
        "opinion_id": "...",
        "text": "...",
        "similarity": 0.73,
        "predicted_type": "Evidence",
        "confidence": 0.61,
        "probs": {"Claim":0.1, "Evidence":0.61, ...}
      }
    ]
  }
}
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.opinion_classifier import OpinionClassifier


def load_matching_results(input_path: Path):
    """Load the topic-to-opinions matching results."""
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded matching results for {len(results)} topics")
    return results


def load_topics_data(data_dir: Path):
    """Load topics data to get topic texts."""
    topics_df = pd.read_csv(data_dir / "topics.csv")

    # Create topic_id to text mapping
    topic_id_to_text = dict(zip(
        topics_df["topic_id"].astype(str),
        topics_df["text"].astype(str)
    ))

    print(f"Loaded topic texts for {len(topic_id_to_text)} topics")
    return topic_id_to_text


def classify_opinions(
    matching_results: dict,
    topic_id_to_text: dict,
    classifier: OpinionClassifier,
    batch_size: int = 32,
):
    """
    Classify all matched opinions and organize by topic.

    Args:
        matching_results: Dict mapping topic_id -> list of matched opinions
        topic_id_to_text: Dict mapping topic_id -> topic text
        classifier: Trained OpinionClassifier instance
        batch_size: Batch size for classification

    Returns:
        Dictionary with labeled opinions organized by topic
    """
    labeled_results = {}

    total_topics = len(matching_results)
    total_opinions = sum(len(opinions) for opinions in matching_results.values())

    print(f"\nClassifying {total_opinions} opinions across {total_topics} topics...")

    processed = 0
    for topic_id, opinions in matching_results.items():
        if not opinions:
            labeled_results[topic_id] = {
                "topic_text": topic_id_to_text.get(topic_id, ""),
                "opinions": []
            }
            continue

        # Get topic text
        topic_text = topic_id_to_text.get(topic_id, "")

        # Extract opinion texts
        opinion_texts = [op["opinion_text"] for op in opinions]

        # Classify in batches
        classified_opinions = []
        for i in range(0, len(opinion_texts), batch_size):
            batch_texts = opinion_texts[i:i+batch_size]
            batch_predictions = classifier.batch_predict(batch_texts)

            # Combine with original data
            for j, pred in enumerate(batch_predictions):
                opinion_idx = i + j
                original_opinion = opinions[opinion_idx]

                classified_opinion = {
                    "opinion_id": original_opinion["opinion_id"],
                    "text": original_opinion["opinion_text"],
                    "similarity": original_opinion["similarity"],
                    "predicted_type": pred["label"],
                    "confidence": pred["confidence"],
                    "probs": pred["probabilities"]
                }

                classified_opinions.append(classified_opinion)

        labeled_results[topic_id] = {
            "topic_text": topic_text,
            "opinions": classified_opinions
        }

        processed += len(opinions)
        if (len(labeled_results) % 100) == 0:
            print(f"  Processed {len(labeled_results)}/{total_topics} topics, {processed}/{total_opinions} opinions")

    print(f"  Completed: {total_topics} topics, {total_opinions} opinions")

    return labeled_results


def save_results(results: dict, output_path: Path):
    """Save labeled results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nLabeled results saved to: {output_path}")

    # Print summary statistics
    total_opinions = sum(len(data["opinions"]) for data in results.values())

    # Count predictions by type
    type_counts = {}
    for data in results.values():
        for opinion in data["opinions"]:
            pred_type = opinion["predicted_type"]
            type_counts[pred_type] = type_counts.get(pred_type, 0) + 1

    print(f"\nSummary:")
    print(f"  Topics processed: {len(results)}")
    print(f"  Total opinions classified: {total_opinions}")
    print(f"\nPredictions by type:")
    for pred_type, count in sorted(type_counts.items()):
        percentage = (count / total_opinions * 100) if total_opinions > 0 else 0
        print(f"    {pred_type}: {count} ({percentage:.1f}%)")


def main(
    input_file: str = None,
    output_file: str = None,
    batch_size: int = 32,
):
    """
    Main pipeline for opinion classification.

    Args:
        input_file: Input JSON file with matching results (default: outputs/topic_to_opinions.json)
        output_file: Output JSON file path (default: outputs/topic_to_opinions_labeled.json)
        batch_size: Batch size for classification (default: 32)
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    input_path = base_dir / (input_file or "outputs/topic_to_opinions.json")
    output_path = base_dir / (output_file or "outputs/topic_to_opinions_labeled.json")

    print("=" * 60)
    print("OPINION CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    # Check if input file exists
    if not input_path.exists():
        print(f"\nError: Input file not found: {input_path}")
        print("Please run pipeline/run_matching.py first to generate matching results.")
        return

    # Load classifier
    print("\nLoading opinion classifier...")
    classifier = OpinionClassifier()

    if classifier.is_trained():
        print("  Using trained classifier from models/trained_classifier")
    else:
        print("  Warning: Using base model (not fine-tuned)")

    # Load data
    matching_results = load_matching_results(input_path)
    topic_id_to_text = load_topics_data(data_dir)

    # Classify opinions
    labeled_results = classify_opinions(
        matching_results,
        topic_id_to_text,
        classifier,
        batch_size=batch_size,
    )

    # Save results
    save_results(labeled_results, output_path)

    print("\nClassification pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify matched opinions into Claim/Evidence/Counterclaim/Rebuttal"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON file with matching results (default: outputs/topic_to_opinions.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: outputs/topic_to_opinions_labeled.json)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for classification (default: 32)",
    )

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
    )
