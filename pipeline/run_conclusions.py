"""
CONCLUSION GENERATION PIPELINE

This script generates conclusions for topics using labeled opinions.
Uses OpenAI LLM to synthesize a conclusion from grouped opinions.

Input: outputs/topic_to_opinions_labeled.json
Output: outputs/conclusions_generated.csv

Columns:
- topic_id
- generated_conclusion
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.conclusion_generator import ConclusionGenerator


def load_labeled_results(input_path: Path):
    """Load the labeled topic-opinion results."""
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded labeled results for {len(results)} topics")
    return results


def prepare_opinions_for_generator(labeled_opinions: list) -> list:
    """
    Convert labeled opinions to format expected by ConclusionGenerator.

    ConclusionGenerator expects: [{"text": "...", "type": "Claim"}, ...]
    Our data has: [{"text": "...", "predicted_type": "Evidence", ...}, ...]
    """
    formatted_opinions = []

    for opinion in labeled_opinions:
        formatted_opinions.append({
            "text": opinion["text"],
            "type": opinion["predicted_type"]
        })

    return formatted_opinions


def generate_conclusions(
    labeled_results: dict,
    generator: ConclusionGenerator,
    max_topics: int = None,
    sleep_between_calls: float = 0.5,
):
    """
    Generate conclusions for all topics.

    Args:
        labeled_results: Dict with topic_id -> {topic_text, opinions}
        generator: ConclusionGenerator instance
        max_topics: Limit number of topics (for demo/testing)
        sleep_between_calls: Sleep duration between API calls (rate limiting)

    Returns:
        List of dicts with topic_id and generated_conclusion
    """
    results = []

    topic_ids = list(labeled_results.keys())
    if max_topics:
        topic_ids = topic_ids[:max_topics]
        print(f"Processing first {max_topics} topics (demo mode)")

    total = len(topic_ids)
    print(f"\nGenerating conclusions for {total} topics...")

    for i, topic_id in enumerate(topic_ids, 1):
        data = labeled_results[topic_id]
        topic_text = data["topic_text"]
        labeled_opinions = data["opinions"]

        # Skip if no opinions
        if not labeled_opinions:
            print(f"  [{i}/{total}] Topic {topic_id}: No opinions, skipping")
            results.append({
                "topic_id": topic_id,
                "generated_conclusion": ""
            })
            continue

        # Format opinions for generator
        formatted_opinions = prepare_opinions_for_generator(labeled_opinions)

        # Generate conclusion
        try:
            conclusion = generator.generate(
                topic_text=topic_text,
                opinions=formatted_opinions
            )

            results.append({
                "topic_id": topic_id,
                "generated_conclusion": conclusion
            })

            if i % 10 == 0:
                print(f"  [{i}/{total}] Generated conclusions")

            # Sleep to avoid rate limits
            if sleep_between_calls > 0:
                time.sleep(sleep_between_calls)

        except Exception as e:
            print(f"  [{i}/{total}] Error generating conclusion for topic {topic_id}: {e}")
            results.append({
                "topic_id": topic_id,
                "generated_conclusion": ""
            })

    print(f"  Completed: {total} topics")

    return results


def save_results(results: list, output_path: Path):
    """Save generated conclusions to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    non_empty = df["generated_conclusion"].str.len() > 0
    num_generated = non_empty.sum()

    print(f"\nSummary:")
    print(f"  Total topics: {len(df)}")
    print(f"  Successfully generated: {num_generated}")
    print(f"  Failed/skipped: {len(df) - num_generated}")


def main(
    input_file: str = None,
    output_file: str = None,
    max_topics: int = None,
    sleep_between_calls: float = 0.5,
    api_key: str = None,
):
    """
    Main pipeline for conclusion generation.

    Args:
        input_file: Input JSON file with labeled results (default: outputs/topic_to_opinions_labeled.json)
        output_file: Output CSV file path (default: outputs/conclusions_generated.csv)
        max_topics: Limit number of topics (for demo/testing)
        sleep_between_calls: Sleep duration between API calls (default: 0.5s)
        api_key: OpenAI API key (optional, uses environment variable if not provided)
    """
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / (input_file or "outputs/topic_to_opinions_labeled.json")
    output_path = base_dir / (output_file or "outputs/conclusions_generated.csv")

    print("=" * 60)
    print("CONCLUSION GENERATION PIPELINE")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max topics: {max_topics or 'all'}")
    print(f"Sleep between calls: {sleep_between_calls}s")
    print("=" * 60)

    # Check if input file exists
    if not input_path.exists():
        print(f"\nError: Input file not found: {input_path}")
        print("Please run pipeline/run_classification.py first to generate labeled results.")
        return

    # Initialize conclusion generator
    print("\nInitializing conclusion generator...")
    try:
        generator = ConclusionGenerator(api_key=api_key)
        generator.initialize()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\nOr run with --no_llm flag to skip conclusion generation")
        return

    # Load data
    labeled_results = load_labeled_results(input_path)

    # Generate conclusions
    results = generate_conclusions(
        labeled_results,
        generator,
        max_topics=max_topics,
        sleep_between_calls=sleep_between_calls,
    )

    # Save results
    save_results(results, output_path)

    print("\nConclusion generation pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate conclusions for topics using LLM"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSON file with labeled results (default: outputs/topic_to_opinions_labeled.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: outputs/conclusions_generated.csv)",
    )
    parser.add_argument(
        "--max_topics",
        type=int,
        default=None,
        help="Limit number of topics to process (for demo/testing)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep duration between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional, uses environment variable if not provided)",
    )

    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        max_topics=args.max_topics,
        sleep_between_calls=args.sleep,
        api_key=args.api_key,
    )
