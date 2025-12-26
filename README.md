# NewMind - Social Media Opinion Analysis System

A complete ML-powered pipeline for analyzing social media opinions, matching them to topics, classifying argument types, and generating conclusions using LLM.

## 🎯 Overview

This system implements a three-stage analysis pipeline that runs via CLI:

1. **Topic-Opinion Matching** - Match opinions to topics using embedding similarity (sentence-transformers)
2. **Opinion Classification** - Classify opinions as Claim/Evidence/Counterclaim/Rebuttal (DistilBERT)
3. **Conclusion Generation** - Generate summaries using OpenAI GPT-4o-mini

**Important Architecture Note:**
- **No HTTP layer** - Pipeline runs via CLI commands
- topic_id is used **ONLY for evaluation**, not as a matching feature
- Kafka/gRPC integration planned for future (event-driven architecture)

## 📁 Project Structure

```
NewMind/
├── run_pipeline.py           # Main entry point - run full pipeline
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (OpenAI API key)
│
├── pipeline/                 # Pipeline stages (CLI scripts)
│   ├── run_matching.py       # Stage 1: Topic-opinion matching
│   ├── run_classification.py # Stage 2: Opinion classification
│   └── run_conclusions.py    # Stage 3: Conclusion generation
│
├── models/                   # ML model implementations
│   ├── topic_matcher.py      # Embedding-based topic matching
│   ├── opinion_classifier.py # DistilBERT opinion classifier
│   ├── conclusion_generator.py # OpenAI conclusion generator
│   └── trained_classifier/   # Trained model checkpoint (255MB)
│
├── evaluation/               # Evaluation framework
│   └── evaluate_all.py       # Comprehensive metrics
│
├── data/                     # Input datasets
│   ├── topics.csv            # 4,024 topics
│   ├── opinions.csv          # 27,099 opinions
│   └── conclusions.csv       # 3,351 reference conclusions
│
├── outputs/                  # Pipeline outputs (generated)
│   ├── topic_to_opinions.json
│   ├── topic_to_opinions_labeled.json
│   └── conclusions_generated.csv
│
└── evaluation_results/       # Evaluation metrics (generated)
    ├── matching_metrics.json
    ├── classifier_metrics.json
    └── conclusion_metrics.json
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set OpenAI API Key (Optional - only needed for conclusion generation)

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key-here
```

Or export as environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Run the Full Pipeline

**Option A: Full pipeline with all stages**

```bash
python run_pipeline.py
```

**Option B: Skip LLM conclusion generation (no API key needed)**

```bash
python run_pipeline.py --no_llm
```

**Option C: Demo mode (first 50 topics only)**

```bash
python run_pipeline.py --max_topics 50 --no_llm
```

## 📊 Pipeline Stages

### Stage 1: Topic-Opinion Matching

Matches topics to relevant opinions using embedding similarity (cosine similarity on sentence-transformers embeddings).

**Important:** topic_id is NOT used as a feature during matching - only text embeddings are used.

```bash
# Run standalone
python pipeline/run_matching.py --top_k 10

# With custom parameters
python pipeline/run_matching.py --top_k 20 --max_topics 100
```

**Output:** `outputs/topic_to_opinions.json`

```json
{
  "<topic_id>": [
    {
      "opinion_id": "123",
      "opinion_text": "...",
      "similarity": 0.73
    }
  ]
}
```

### Stage 2: Opinion Classification

Classifies matched opinions into 4 types using trained DistilBERT model:
- **Claim** - Supporting argument
- **Evidence** - Supporting evidence
- **Counterclaim** - Opposing argument
- **Rebuttal** - Response to counterclaim

```bash
# Run standalone (requires Stage 1 output)
python pipeline/run_classification.py
```

**Output:** `outputs/topic_to_opinions_labeled.json`

```json
{
  "<topic_id>": {
    "topic_text": "...",
    "opinions": [
      {
        "opinion_id": "123",
        "text": "...",
        "similarity": 0.73,
        "predicted_type": "Evidence",
        "confidence": 0.61,
        "probs": {
          "Claim": 0.1,
          "Evidence": 0.61,
          "Counterclaim": 0.2,
          "Rebuttal": 0.09
        }
      }
    ]
  }
}
```

### Stage 3: Conclusion Generation

Generates conclusion summaries using OpenAI GPT-4o-mini.

**Requires:** OpenAI API key in environment

```bash
# Run standalone (requires Stage 2 output)
python pipeline/run_conclusions.py --max_topics 50 --sleep 0.5

# With custom API key
python pipeline/run_conclusions.py --api_key "sk-..."
```

**Output:** `outputs/conclusions_generated.csv`

| topic_id | generated_conclusion |
|----------|---------------------|
| topic_1  | Based on the evidence... |

### Stage 4: Evaluation

Evaluates all pipeline components against ground truth data.

```bash
# Run standalone
python evaluation/evaluate_all.py

# With BERTScore (slower but more semantic)
python evaluation/evaluate_all.py --use_bertscore

# Evaluate specific component only
python evaluation/evaluate_all.py --matching_only
python evaluation/evaluate_all.py --classification_only
python evaluation/evaluate_all.py --conclusions_only
```

**Outputs:**

1. **Matching Metrics** (`evaluation_results/matching_metrics.json`)
   - Recall@k: Proportion of relevant opinions retrieved
   - Precision@k: Proportion of retrieved opinions that are relevant
   - MRR: Mean Reciprocal Rank

2. **Classification Metrics** (`evaluation_results/classifier_metrics.json`)
   - Macro F1, Weighted F1, Accuracy
   - Per-class F1, Precision, Recall
   - Confusion matrix

3. **Conclusion Metrics** (`evaluation_results/conclusion_metrics.json`)
   - ROUGE-1, ROUGE-2, ROUGE-L F1 scores
   - BERTScore (if enabled)

## 🔧 Advanced Usage

### Custom Pipeline Parameters

```bash
# Use top-20 matches instead of top-10
python run_pipeline.py --top_k 20

# Set similarity threshold
python run_pipeline.py --threshold 0.6

# Use relative margin filtering
python run_pipeline.py --relative_margin 0.05

# Combine parameters
python run_pipeline.py --top_k 15 --max_topics 100 --no_llm
```

### Individual Pipeline Stages

Run stages independently for debugging or custom workflows:

```bash
# 1. Matching only
python pipeline/run_matching.py --top_k 10 --output outputs/custom_matching.json

# 2. Classification only
python pipeline/run_classification.py --input outputs/custom_matching.json

# 3. Conclusions only
python pipeline/run_conclusions.py --max_topics 10 --sleep 1.0

# 4. Evaluation only
python evaluation/evaluate_all.py --top_k 10
```

## 📈 Training Models (Optional)

The repository includes a pre-trained classifier in `models/trained_classifier/`. If you want to retrain:

### Train Opinion Classifier

```bash
# Quick training (2K samples, ~2 minutes)
python quick_train.py

# Full training (21K samples, ~30 minutes)
python train_and_evaluate.py
```

**Expected Performance:**
- Accuracy: ~75%
- Macro F1: ~40% (due to class imbalance)
- Weighted F1: ~71%
- Claim/Evidence: ~78-81% F1
- Counterclaim/Rebuttal: Lower due to fewer samples

## 📊 Dataset Statistics

| File | Records | Description |
|------|---------|-------------|
| topics.csv | 4,024 | Unique topics with position statements |
| opinions.csv | 27,099 | Opinions linked to topics with types |
| conclusions.csv | 3,351 | Reference conclusion summaries |

**Opinion Type Distribution:**
- Evidence: 9,675 (35.7%)
- Claim: 9,574 (35.3%)
- Counterclaim: 1,411 (5.2%)
- Rebuttal: 1,000 (3.7%)

## ⚙️ Configuration

All settings are in `config/settings.py`:

```python
# Model settings
TOPIC_MATCHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_MODEL = "distilbert-base-uncased"
OPENAI_MODEL = "gpt-4o-mini"

# Training settings
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 256
```

## 🔍 Important Notes

### Why topic_id is NOT used as a feature

The system uses **only text embeddings** for matching, not topic_id:
- ✅ Enables matching new opinions to any topic
- ✅ Works with unseen topics
- ✅ Real-world applicable (no pre-existing topic assignments)
- ❌ topic_id is only used for evaluation/validation

### Architecture Design

Current implementation is **CLI-based** for simplicity and reproducibility:
- Direct file I/O (JSON/CSV)
- No HTTP overhead
- Easy to debug and test

**Future work:**
- Event-driven architecture with Kafka
- gRPC API for service integration
- Real-time streaming pipeline

## 🧪 Testing

Run the test suite:

```bash
python test_system.py
```

Expected output: All tests passing ✅

## 🚧 Troubleshooting

**Issue:** NumPy compatibility error

```bash
pip install "numpy<2.0"
```

**Issue:** OpenAI API key not found

```bash
export OPENAI_API_KEY='your-key-here'
# Or use --no_llm flag to skip conclusion generation
```

**Issue:** Out of memory during classification

```bash
# Reduce batch size in models/opinion_classifier.py
# Or process fewer topics: --max_topics 100
```

## 📝 License

Internal project for DigitalPulse social media analysis.

## 🤝 Contributing

For questions or issues, please contact the development team.

---

**Version:** 2.0.0 (CLI Pipeline)
**Last Updated:** 2025-12-26
