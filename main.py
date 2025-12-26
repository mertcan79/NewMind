#!/usr/bin/env python3
"""
MAIN ENTRY POINT FOR SOCIAL MEDIA ANALYSIS SYSTEM
Provides functions for training, evaluation, and running services.
No argparse - call functions directly.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def train_classifier(
    topics_path: str = None,
    opinions_path: str = None,
    conclusions_path: str = None,
    output_dir: str = "trained_models",
    epochs: int = 3,
    batch_size: int = 16
):
    """
    TRAIN THE OPINION CLASSIFIER

    Args:
        topics_path: Path to topics.csv (default: data/topics.csv)
        opinions_path: Path to opinions.csv (default: data/opinions.csv)
        conclusions_path: Path to conclusions.csv (default: data/conclusions.csv)
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Evaluation results dict
    """
    from data import DataProcessor
    from models import OpinionClassifier
    from config.settings import settings

    # SET DEFAULT PATHS
    topics_path = topics_path or str(settings.DATA_DIR / "topics.csv")
    opinions_path = opinions_path or str(settings.DATA_DIR / "opinions.csv")
    conclusions_path = conclusions_path or str(settings.DATA_DIR / "conclusions.csv")

    print("=" * 60)
    print("TRAINING OPINION CLASSIFIER")
    print("=" * 60)

    # LOAD DATA
    processor = DataProcessor()
    processor.load_data(
        topics_path=topics_path,
        opinions_path=opinions_path,
        conclusions_path=conclusions_path
    )

    # PREPARE DATA
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        processor.prepare_classification_data()

    # INITIALIZE CLASSIFIER
    classifier = OpinionClassifier()
    classifier.load_model()

    # TRAIN
    save_path = Path(output_dir) / "classifier"
    history = classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path
    )

    # EVALUATE ON TEST SET
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    results = classifier.evaluate(test_texts, test_labels)

    print(f"\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print("\n  Per-class F1:")
    for label, f1 in results["f1_per_class"].items():
        print(f"    {label}: {f1:.4f}")

    return results


def train_topic_matcher(
    topics_path: str = None,
    opinions_path: str = None,
    conclusions_path: str = None,
    output_dir: str = "trained_models"
):
    """
    PREPARE AND SAVE TOPIC EMBEDDINGS

    Args:
        topics_path: Path to topics.csv
        opinions_path: Path to opinions.csv
        conclusions_path: Path to conclusions.csv
        output_dir: Directory to save topic matcher
    """
    from data import DataProcessor
    from models import TopicMatcher
    from config.settings import settings

    # SET DEFAULT PATHS
    topics_path = topics_path or str(settings.DATA_DIR / "topics.csv")
    opinions_path = opinions_path or str(settings.DATA_DIR / "opinions.csv")
    conclusions_path = conclusions_path or str(settings.DATA_DIR / "conclusions.csv")

    print("=" * 60)
    print("PREPARING TOPIC MATCHER")
    print("=" * 60)

    # LOAD DATA
    processor = DataProcessor()
    processor.load_data(
        topics_path=topics_path,
        opinions_path=opinions_path,
        conclusions_path=conclusions_path
    )

    # PREPARE DATA
    topics, _ = processor.prepare_topic_matching_data()

    # INITIALIZE AND ENCODE
    matcher = TopicMatcher()
    matcher.load_model()
    matcher.encode_topics(
        topic_ids=topics["topic_id"].tolist(),
        topic_texts=topics["clean_text"].tolist()
    )

    # SAVE
    save_path = Path(output_dir) / "topic_matcher"
    matcher.save(save_path)

    print(f"\nTopic matcher saved to {save_path}")
    return matcher


def evaluate_all(
    topics_path: str = None,
    opinions_path: str = None,
    conclusions_path: str = None,
    sample_size: int = 500
):
    """
    RUN EVALUATION ON ALL COMPONENTS

    Args:
        topics_path: Path to topics.csv
        opinions_path: Path to opinions.csv
        conclusions_path: Path to conclusions.csv
        sample_size: Number of samples to evaluate on
    """
    from evaluation import Evaluator
    from data import DataProcessor
    from models import OpinionClassifier
    from config.settings import settings

    # SET DEFAULT PATHS
    topics_path = topics_path or str(settings.DATA_DIR / "topics.csv")
    opinions_path = opinions_path or str(settings.DATA_DIR / "opinions.csv")
    conclusions_path = conclusions_path or str(settings.DATA_DIR / "conclusions.csv")

    print("=" * 60)
    print("RUNNING EVALUATION PIPELINE")
    print("=" * 60)

    # LOAD DATA
    processor = DataProcessor()
    processor.load_data(
        topics_path=topics_path,
        opinions_path=opinions_path,
        conclusions_path=conclusions_path
    )

    # INITIALIZE EVALUATOR
    evaluator = Evaluator()

    # LOAD CLASSIFIER
    classifier = OpinionClassifier()
    classifier.load_model()

    # GET TEST DATA
    _, _, _, _, test_texts, test_labels = processor.prepare_classification_data()

    # EVALUATE
    predictions = classifier.predict(test_texts[:sample_size])
    results = evaluator.evaluate_classification(
        predicted_labels=predictions,
        true_labels=test_labels[:sample_size]
    )

    # PRINT RESULTS
    evaluator.print_results({
        "components": {"classification": results}
    })

    return results


def run_grpc(
    host: str = "localhost",
    port: int = 50051,
    classifier_path: str = None,
    topic_matcher_path: str = None,
    enable_conclusion: bool = False
):
    """
    RUN THE GRPC SERVER

    Args:
        host: Server host
        port: Server port
        classifier_path: Path to trained classifier
        topic_matcher_path: Path to trained topic matcher
        enable_conclusion: Enable conclusion generation (requires OpenAI key)
    """
    from grpc_service import serve
    from models import TopicMatcher, OpinionClassifier, ConclusionGenerator

    # LOAD MODELS IF PATHS PROVIDED
    topic_matcher = None
    classifier = None
    conclusion_generator = None

    if classifier_path:
        classifier = OpinionClassifier()
        classifier.load(Path(classifier_path))

    if topic_matcher_path:
        topic_matcher = TopicMatcher()
        topic_matcher.load(Path(topic_matcher_path))

    if enable_conclusion:
        conclusion_generator = ConclusionGenerator()
        conclusion_generator.initialize()

    serve(
        topic_matcher=topic_matcher,
        classifier=classifier,
        conclusion_generator=conclusion_generator,
        host=host,
        port=port
    )


def run_worker(classifier_path: str = None, enable_conclusion: bool = False):
    """
    RUN THE REDIS EVENT WORKER

    Args:
        classifier_path: Path to trained classifier
        enable_conclusion: Enable conclusion generation
    """
    from events import EventConsumer
    from models import OpinionClassifier, ConclusionGenerator

    consumer = EventConsumer()
    consumer.connect()

    # LOAD MODELS
    classifier = None
    conclusion_generator = None

    if classifier_path:
        classifier = OpinionClassifier()
        classifier.load(Path(classifier_path))

    if enable_conclusion:
        conclusion_generator = ConclusionGenerator()
        conclusion_generator.initialize()

    consumer.initialize_models(
        classifier=classifier,
        conclusion_generator=conclusion_generator
    )

    consumer.run()


def compile_protos():
    """COMPILE PROTOBUF FILES"""
    import subprocess

    proto_path = Path("protos")
    output_path = Path("grpc_service")

    cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"-I{proto_path}",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
        str(proto_path / "opinion_service.proto")
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Protobuf compilation successful!")

        # FIX IMPORTS IN GENERATED FILE
        grpc_file = output_path / "opinion_service_pb2_grpc.py"
        if grpc_file.exists():
            content = grpc_file.read_text()
            content = content.replace(
                "import opinion_service_pb2",
                "from grpc_service import opinion_service_pb2"
            )
            grpc_file.write_text(content)
            print("Fixed imports in generated files")
    else:
        print(f"Error: {result.stderr}")


if __name__ == "__main__":
    # EXAMPLE USAGE - UNCOMMENT THE FUNCTION YOU WANT TO RUN

    # TRAIN CLASSIFIER
    # train_classifier(epochs=3)

    # TRAIN TOPIC MATCHER
    # train_topic_matcher()

    # EVALUATE
    # evaluate_all()

    # RUN API SERVER
    # run_api(port=8000)

    # RUN GRPC SERVER
    # run_grpc(port=50051)

    # COMPILE PROTOS
    # compile_protos()

    print("Social Media Analysis System")
    print("=" * 40)
    print("Available functions:")
    print("  - train_classifier()")
    print("  - train_topic_matcher()")
    print("  - evaluate_all()")
    print("  - run_api()")
    print("  - run_grpc()")
    print("  - run_worker()")
    print("  - compile_protos()")
    print("\nImport and call directly or use solution.py")
