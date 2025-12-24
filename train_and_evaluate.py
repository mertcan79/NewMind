#!/usr/bin/env python3
"""
TRAINING AND EVALUATION SCRIPT
Trains the opinion classifier and saves comprehensive evaluation results
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
print("OPINION CLASSIFIER TRAINING AND EVALUATION")
print("=" * 70)

# STEP 1: LOAD DATA
print("\n[STEP 1/5] Loading data...")
processor = DataProcessor()
processor.load_data()

# GET TRAIN/VAL/TEST SPLITS
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    processor.prepare_classification_data()

print(f"  Training samples: {len(train_texts)}")
print(f"  Validation samples: {len(val_texts)}")
print(f"  Test samples: {len(test_texts)}")

# STEP 2: INITIALIZE CLASSIFIER
print("\n[STEP 2/5] Initializing classifier...")
classifier = OpinionClassifier()
classifier.load_model()

# STEP 3: TRAIN THE MODEL
print("\n[STEP 3/5] Training the model...")
print("  This will take several minutes...")
print("  Training for 3 epochs with batch size 16")

output_dir = Path("trained_models/classifier")
output_dir.mkdir(parents=True, exist_ok=True)

history = classifier.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    epochs=3,
    batch_size=16,
    save_path=output_dir
)

print("\n  Training completed!")
print(f"  Best model saved to: {output_dir}")

# STEP 4: EVALUATE ON TEST SET
print("\n[STEP 4/5] Evaluating on test set...")
predictions = classifier.predict(test_texts)

evaluator = Evaluator(output_dir=Path("evaluation_results"))
results = evaluator.evaluate_classification(
    predicted_labels=predictions,
    true_labels=test_labels
)

# STEP 5: SAVE RESULTS
print("\n[STEP 5/5] Saving evaluation results...")

# COMPREHENSIVE RESULTS
full_results = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": {
        "name": "distilbert-base-uncased",
        "fine_tuned": True,
        "num_labels": 4,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "dataset": {
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "test_samples": len(test_texts),
        "total_samples": len(train_texts) + len(val_texts) + len(test_texts)
    },
    "training_history": {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_f1": history["val_f1"],
        "val_accuracy": history["val_accuracy"]
    },
    "test_evaluation": results
}

# SAVE TO FILE
output_file = evaluator.save_results(full_results, name="classifier_evaluation")

# PRINT SUMMARY
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(f"\nOverall Performance:")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  F1 (weighted): {results['f1']['weighted']:.4f}")
print(f"  F1 (macro): {results['f1']['macro']:.4f}")
print(f"  F1 (micro): {results['f1']['micro']:.4f}")

print(f"\nPer-Class F1 Scores:")
for label, f1 in results['f1']['per_class'].items():
    print(f"  {label:15s}: {f1:.4f}")

print(f"\nPer-Class Precision:")
for label, prec in results['precision']['per_class'].items():
    print(f"  {label:15s}: {prec:.4f}")

print(f"\nPer-Class Recall:")
for label, rec in results['recall']['per_class'].items():
    print(f"  {label:15s}: {rec:.4f}")

print(f"\nClass Distribution in Test Set:")
for label, count in results['class_distribution'].items():
    percentage = (count / results['num_samples']) * 100
    print(f"  {label:15s}: {count:5d} ({percentage:5.1f}%)")

print(f"\nConfusion Matrix:")
print("  Predicted →")
print("  Actual ↓")
import numpy as np
conf_matrix = np.array(results['confusion_matrix'])
labels = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
print(f"            {'  '.join([f'{l[:5]:>5s}' for l in labels])}")
for i, label in enumerate(labels):
    row = conf_matrix[i]
    print(f"  {label[:5]:>5s}:  {'  '.join([f'{int(v):>5d}' for v in row])}")

print(f"\nTraining History:")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
print(f"  Final val F1: {history['val_f1'][-1]:.4f}")
print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")

print("\n" + "=" * 70)
print("TRAINING AND EVALUATION COMPLETE!")
print("=" * 70)
print(f"\nModel saved to: {output_dir}")
print(f"Results saved to: {output_file}")
print("\nTo use the trained model:")
print("  from models import OpinionClassifier")
print("  classifier = OpinionClassifier()")
print(f"  classifier.load(Path('{output_dir}'))")
print("  result = classifier.predict_single('Your opinion text here')")
