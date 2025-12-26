"""
FULL PIPELINE RUNNER - ONE COMMAND EXECUTION

This script runs the complete social media analysis pipeline:
1. Topic-Opinion Matching (embedding similarity)
2. Opinion Classification (DistilBERT)
3. Conclusion Generation (LLM - optional)
4. Evaluation (all metrics)

Usage:
    # Run full pipeline with default settings
    python run_pipeline.py

    # Run with custom parameters
    python run_pipeline.py --top_k 10 --max_topics 50

    # Skip LLM conclusion generation
    python run_pipeline.py --no_llm

    # Run with BERTScore evaluation
    python run_pipeline.py --use_bertscore

Outputs:
    outputs/topic_to_opinions.json
    outputs/topic_to_opinions_labeled.json
    outputs/conclusions_generated.csv (if --no_llm not set)
    evaluation_results/matching_metrics.json
    evaluation_results/classifier_metrics.json
    evaluation_results/conclusion_metrics.json (if --no_llm not set)
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import pipeline modules
from pipeline import run_matching, run_classification, run_conclusions
from evaluation import evaluate_all


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_full_pipeline(
    top_k: int = 10,
    max_topics: int = None,
    no_llm: bool = False,
    use_bertscore: bool = False,
    threshold: float = None,
    relative_margin: float = None,
    sleep_between_calls: float = 0.5,
    api_key: str = None,
):
    """
    Run the complete pipeline.

    Args:
        top_k: Number of top opinions to retrieve per topic
        max_topics: Limit number of topics to process (for demo/testing)
        no_llm: Skip LLM conclusion generation
        use_bertscore: Use BERTScore for conclusion evaluation
        threshold: Minimum similarity threshold for matching
        relative_margin: Relative margin for matching
        sleep_between_calls: Sleep between LLM API calls
        api_key: OpenAI API key (optional)
    """
    print_header("NEWMIND SOCIAL MEDIA ANALYSIS PIPELINE")
    print(f"\nConfiguration:")
    print(f"  Top-k: {top_k}")
    print(f"  Max topics: {max_topics or 'all'}")
    print(f"  Skip LLM: {no_llm}")
    print(f"  BERTScore: {use_bertscore}")
    print(f"  Similarity threshold: {threshold or 'default'}")
    print(f"  Relative margin: {relative_margin or 'none'}")

    try:
        # ====================================================================
        # STAGE 1: TOPIC-OPINION MATCHING
        # ====================================================================
        print_header("STAGE 1: TOPIC-OPINION MATCHING")
        print("Matching topics to opinions using embedding similarity...")
        print("NOTE: topic_id is NOT used as a feature, only for output organization")

        run_matching.main(
            top_k=top_k,
            threshold=threshold,
            relative_margin=relative_margin,
            max_topics=max_topics,
        )

        # ====================================================================
        # STAGE 2: OPINION CLASSIFICATION
        # ====================================================================
        print_header("STAGE 2: OPINION CLASSIFICATION")
        print("Classifying opinions into Claim/Evidence/Counterclaim/Rebuttal...")

        run_classification.main()

        # ====================================================================
        # STAGE 3: CONCLUSION GENERATION (OPTIONAL)
        # ====================================================================
        if not no_llm:
            print_header("STAGE 3: CONCLUSION GENERATION")
            print("Generating conclusions using LLM...")

            try:
                run_conclusions.main(
                    max_topics=max_topics,
                    sleep_between_calls=sleep_between_calls,
                    api_key=api_key,
                )
            except Exception as e:
                print(f"\nWarning: Conclusion generation failed: {e}")
                print("Continuing with evaluation (conclusions will be skipped)...")
                no_llm = True
        else:
            print_header("STAGE 3: CONCLUSION GENERATION")
            print("Skipped (--no_llm flag set)")

        # ====================================================================
        # STAGE 4: EVALUATION
        # ====================================================================
        print_header("STAGE 4: EVALUATION")
        print("Evaluating all pipeline components...")

        # Determine which evaluations to run
        if no_llm:
            print("NOTE: Skipping conclusion evaluation (no LLM output)")

        evaluate_all.main(
            top_k=top_k,
            use_bertscore=use_bertscore if not no_llm else False,
        )

        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        print_header("PIPELINE COMPLETE!")

        print("\nOutput files:")
        print("  outputs/topic_to_opinions.json")
        print("  outputs/topic_to_opinions_labeled.json")
        if not no_llm:
            print("  outputs/conclusions_generated.csv")

        print("\nEvaluation results:")
        print("  evaluation_results/matching_metrics.json")
        print("  evaluation_results/classifier_metrics.json")
        if not no_llm:
            print("  evaluation_results/conclusion_metrics.json")

        print("\n" + "=" * 70)
        print("SUCCESS! All pipeline stages completed.")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Pipeline failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the complete NewMind social media analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python run_pipeline.py

  # Run on first 50 topics (demo mode)
  python run_pipeline.py --max_topics 50

  # Skip LLM conclusion generation (faster, no API key needed)
  python run_pipeline.py --no_llm

  # Use top-20 matches instead of top-10
  python run_pipeline.py --top_k 20

  # Enable BERTScore for conclusion evaluation (slower but more accurate)
  python run_pipeline.py --use_bertscore

Note:
  - topic_id is used ONLY for evaluation, not as a matching feature
  - Architecture is designed for event-driven + gRPC integration (future work)
        """
    )

    # Matching parameters
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
        help="Minimum similarity threshold for matching (optional)",
    )
    parser.add_argument(
        "--relative_margin",
        type=float,
        default=None,
        help="Relative margin from best score for matching (optional)",
    )

    # General parameters
    parser.add_argument(
        "--max_topics",
        type=int,
        default=None,
        help="Limit number of topics to process (for demo/testing)",
    )

    # LLM parameters
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip LLM conclusion generation (no API key needed)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional, uses environment variable if not provided)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep duration between LLM API calls in seconds (default: 0.5)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--use_bertscore",
        action="store_true",
        help="Use BERTScore for conclusion evaluation (slower but more semantic)",
    )

    args = parser.parse_args()

    # Run pipeline
    run_full_pipeline(
        top_k=args.top_k,
        max_topics=args.max_topics,
        no_llm=args.no_llm,
        use_bertscore=args.use_bertscore,
        threshold=args.threshold,
        relative_margin=args.relative_margin,
        sleep_between_calls=args.sleep,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
