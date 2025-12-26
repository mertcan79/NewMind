"""
TOPIC TO OPINIONS MATCHING PIPELINE

This script matches topics to relevant opinions using embedding similarity.
IMPORTANT: topic_id is NOT used as a feature during matching, only for output organization.

Output format:
{
  "<topic_id>": [
    {"opinion_id": "...", "similarity": 0.73, "text": "..."},
    ...
  ]
}
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.topic_matcher import TopicMatcher


def load_data(data_dir: Path):
    """Load topics and opinions from CSV files."""
    topics_df = pd.read_csv(data_dir / "topics.csv")
    opinions_df = pd.read_csv(data_dir / "opinions.csv")

    print(f"Loaded {len(topics_df)} topics")
    print(f"Loaded {len(opinions_df)} opinions")

    return topics_df, opinions_df


def run_matching(
    topics_df: pd.DataFrame,
    opinions_df: pd.DataFrame,
    top_k: int = 10,
    threshold: float = None,
    relative_margin: float = None,
    max_topics: int = None,
):
    """
    Match topics to opinions using embedding similarity.

    Args:
        topics_df: DataFrame with columns [id, topic_id, text, type, effectiveness]
        opinions_df: DataFrame with columns [id, topic_id, text, type, effectiveness]
        top_k: Number of top opinions to retrieve per topic
        threshold: Minimum similarity threshold (optional)
        relative_margin: Relative margin from best score (optional)
        max_topics: Limit number of topics to process (for testing)

    Returns:
        Dictionary mapping topic_id -> list of matched opinions
    """
    # Initialize matcher
    matcher = TopicMatcher()

    # Prepare data
    if max_topics:
        topics_df = topics_df.head(max_topics)
        print(f"Processing first {max_topics} topics (demo mode)")

    # Extract unique topics
    # Note: We use topic_id only as an identifier, NOT as a feature
    topic_ids = topics_df["topic_id"].astype(str).tolist()
    topic_texts = topics_df["text"].astype(str).tolist()

    # Extract opinions
    opinion_ids = opinions_df["id"].astype(str).tolist()
    opinion_texts = opinions_df["text"].astype(str).tolist()

    print("\nEncoding topics...")
    matcher.encode_topics(topic_ids, topic_texts)

    print("Encoding opinions...")
    matcher.encode_opinions(opinion_ids, opinion_texts)

    print("Computing similarity matrix...")
    matcher.compute_similarity_matrix()

    print(f"Matching topics to top-{top_k} opinions...")
    results = matcher.match_all_topics(
        top_k=top_k,
        threshold=threshold,
        relative_margin=relative_margin,
    )

    print(f"Matched {len(results)} topics")

    return results


def save_results(results: dict, output_path: Path):
    """Save matching results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    total_matches = sum(len(opinions) for opinions in results.values())
    avg_matches = total_matches / len(results) if results else 0

    print(f"\nSummary:")
    print(f"  Topics processed: {len(results)}")
    print(f"  Total matches: {total_matches}")
    print(f"  Average matches per topic: {avg_matches:.2f}")


def main(
    top_k: int = 10,
    threshold: float = None,
    relative_margin: float = None,
    max_topics: int = None,
    output_file: str = None,
):
    """
    Main pipeline for topic-to-opinions matching.

    Args:
        top_k: Number of top opinions to retrieve per topic
        threshold: Minimum similarity threshold (optional)
        relative_margin: Relative margin from best score (optional)
        max_topics: Limit number of topics to process (for testing)
        output_file: Output JSON file path (default: outputs/topic_to_opinions.json)
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_path = base_dir / (output_file or "outputs/topic_to_opinions.json")

    print("=" * 60)
    print("TOPIC TO OPINIONS MATCHING PIPELINE")
    print("=" * 60)
    print(f"Top-k: {top_k}")
    print(f"Threshold: {threshold}")
    print(f"Relative margin: {relative_margin}")
    print(f"Max topics: {max_topics or 'all'}")
    print("=" * 60)

    # Load data
    topics_df, opinions_df = load_data(data_dir)

    # Run matching
    results = run_matching(
        topics_df,
        opinions_df,
        top_k=top_k,
        threshold=threshold,
        relative_margin=relative_margin,
        max_topics=max_topics,
    )

    # Save results
    save_results(results, output_path)

    print("\nMatching pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match topics to relevant opinions using embedding similarity"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top opinions to retrieve per topic (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold (optional)",
    )
    parser.add_argument(
        "--relative_margin",
        type=float,
        default=None,
        help="Relative margin from best score (optional)",
    )
    parser.add_argument(
        "--max_topics",
        type=int,
        default=None,
        help="Limit number of topics to process (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: outputs/topic_to_opinions.json)",
    )

    args = parser.parse_args()

    main(
        top_k=args.top_k,
        threshold=args.threshold,
        relative_margin=args.relative_margin,
        max_topics=args.max_topics,
        output_file=args.output,
    )
