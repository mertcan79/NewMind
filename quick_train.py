#!/usr/bin/env python3
"""
QUICK TRAINING SCRIPT
Trains on a subset of data for faster results and verification
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import DataProcessor
from models import OpinionClassifier
from evaluation import Evaluator
import json
from datetime import datetime

print("=" * 70)
print("QUICK TRAINING (SUBSET) FOR VERIFICATION")
print("=" * 70)

# LOAD DATA
print("\n[1/5] Loading data...")
processor = DataProcessor()
processor.load_data()

train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    processor.prepare_classification_data()

# USE SUBSET FOR QUICK TRAINING
TRAIN_SIZE = 2000
VAL_SIZE = 500
TEST_SIZE = 500

print(f"\nUsing subset:")
print(f"  Training: {TRAIN_SIZE} samples (from {len(train_texts)})")
print(f"  Validation: {VAL_SIZE} samples (from {len(val_texts)})")
print(f"  Test: {TEST_SIZE} samples (from {len(test_texts)})")

train_texts = train_texts[:TRAIN_SIZE]
train_labels = train_labels[:TRAIN_SIZE]
val_texts = val_texts[:VAL_SIZE]
val_labels = val_labels[:VAL_SIZE]
test_texts = test_texts[:TEST_SIZE]
test_labels = test_labels[:TEST_SIZE]

# INITIALIZE CLASSIFIER
print("\n[2/5] Initializing classifier...")
classifier = OpinionClassifier()
classifier.load_model()

# TRAIN
print("\n[3/5] Training for 2 epochs...")
output_dir = Path("trained_models/classifier_quick")
output_dir.mkdir(parents=True, exist_ok=True)

history = classifier.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    epochs=2,
    batch_size=16,
    save_path=output_dir
)

print("\nTraining complete!")

# EVALUATE
print("\n[4/5] Evaluating...")
predictions = classifier.predict(test_texts)

evaluator = Evaluator(output_dir=Path("evaluation_results"))
results = evaluator.evaluate_classification(
    predicted_labels=predictions,
    true_labels=test_labels
)

# SAVE RESULTS
print("\n[5/5] Saving results...")
full_results = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": {
        "name": "distilbert-base-uncased",
        "fine_tuned": True,
        "epochs": 2,
        "subset_training": True,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE
    },
    "training_history": history,
    "test_evaluation": results
}

output_file = evaluator.save_results(full_results, name="quick_classifier_evaluation")

# PRINT RESULTS
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nOverall Performance:")
print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
print(f"  F1 (weighted): {results['f1']['weighted']:.4f}")
print(f"  F1 (macro): {results['f1']['macro']:.4f}")

print(f"\nPer-Class Performance:")
labels_list = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
print(f"{'Class':<15} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Support':>10}")
print("-" * 60)
for label in labels_list:
    f1 = results['f1']['per_class'][label]
    prec = results['precision']['per_class'][label]
    rec = results['recall']['per_class'][label]
    support = results['class_distribution'][label]
    print(f"{label:<15} {f1:>8.4f} {prec:>10.4f} {rec:>10.4f} {support:>10d}")

print(f"\nTraining Progress:")
print(f"  Epoch 1 - Val F1: {history['val_f1'][0]:.4f}, Val Acc: {history['val_accuracy'][0]:.4f}")
print(f"  Epoch 2 - Val F1: {history['val_f1'][1]:.4f}, Val Acc: {history['val_accuracy'][1]:.4f}")

# TEST SOME PREDICTIONS
print(f"\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

test_examples = [
    "I think climate change is real because of scientific evidence",
    "Some people disagree with this position",
    "However, the data clearly shows a warming trend",
    "Studies have shown that CO2 levels are rising"
]

for text in test_examples:
    result = classifier.predict_single(text)
    print(f"\nText: {text}")
    print(f"Predicted: {result['predicted_type']}")
    print(f"Confidence: {max(result['probabilities'].values()):.2f}")
    print(f"All probabilities: {result['probabilities']}")

print("\n" + "=" * 70)
print("QUICK TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel saved to: {output_dir}")
print(f"Results saved to: {output_file}")
print("\nNote: This was trained on a subset. For production, train on full dataset.")
