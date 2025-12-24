# Claude Code Handoff: Social Media Analysis Project

## Project Overview

This is an AI Engineer assessment project for DigitalPulse - a social media analysis system that:
1. **Matches opinions to topics** using sentence embeddings
2. **Classifies opinion types** (Claim, Counterclaim, Rebuttal, Evidence) using DistilBERT
3. **Generates conclusions** using OpenAI API

Architecture: FastAPI → Redis (event queue) → gRPC ML Service

---

## Current Project Status

### ✅ COMPLETED FILES

| File | Status | Description |
|------|--------|-------------|
| `config/settings.py` | ✅ Ready | Configuration with Pydantic |
| `config/__init__.py` | ✅ Ready | Exports settings |
| `data/preprocessing.py` | ✅ Ready | Data loading, cleaning, train/val/test splits |
| `data/__init__.py` | ✅ Ready | Exports DataProcessor |
| `models/topic_matcher.py` | ✅ Ready | Sentence-transformer based matching |
| `models/opinion_classifier.py` | ✅ Ready | DistilBERT fine-tuning & inference |
| `models/conclusion_generator.py` | ✅ Ready | OpenAI API integration |
| `models/__init__.py` | ✅ Ready | Exports all models |
| `events/producer.py` | ✅ Ready | Redis event publisher |
| `events/consumer.py` | ✅ Ready | Redis event worker |
| `events/__init__.py` | ✅ Ready | Exports producer/consumer |
| `api/app.py` | ✅ Ready | FastAPI endpoints |
| `api/__init__.py` | ✅ Ready | Exports app |
| `evaluation/evaluator.py` | ✅ Ready | F1, accuracy, ROUGE metrics |
| `evaluation/__init__.py` | ✅ Ready | Exports Evaluator |
| `main.py` | ✅ Ready | CLI entry point |
| `requirements.txt` | ✅ Ready | All dependencies |
| `README.md` | ✅ Ready | Documentation |

### ⚠️ NEEDS CREATION (Proto & gRPC)

| File | Status | Description |
|------|--------|-------------|
| `protos/opinion_service.proto` | ❌ Missing | Protocol buffer definitions |
| `grpc_service/__init__.py` | ❌ Missing | gRPC module init |
| `grpc_service/server.py` | ❌ Missing | gRPC server implementation |
| `grpc_service/client.py` | ❌ Missing | gRPC client for testing |

### 📁 DATA FILES (Should be in project root or data/)

The CSV files should be placed in the project:
- `topics.csv` - 4,024 topics with positions
- `opinions.csv` - 27,099 opinions (Claim, Evidence, Counterclaim, Rebuttal)
- `conclusions.csv` - 3,351 reference conclusions

---

## Quick Test Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Data Loading
```python
from data import DataProcessor

processor = DataProcessor()
processor.load_data(
    topics_path="path/to/topics.csv",
    opinions_path="path/to/opinions.csv",
    conclusions_path="path/to/conclusions.csv"
)

# Analyze data
analysis = processor.analyze_data()
print(analysis)

# Prepare classification data
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
    processor.prepare_classification_data()
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
```

### 3. Test Classifier (No Training - Base Model)
```python
from models import OpinionClassifier

classifier = OpinionClassifier()
classifier.load_model()  # Loads base DistilBERT

# Single prediction
result = classifier.predict_single("I think this is true because of the evidence")
print(result)
# {'predicted_type': 'Evidence', 'predicted_id': 3, 'probabilities': {...}}
```

### 4. Test Topic Matcher
```python
from models import TopicMatcher

matcher = TopicMatcher()
matcher.load_model()

# Encode some topics
matcher.encode_topics(
    topic_ids=["T1", "T2"],
    topic_texts=["Climate change is real", "Mars has no life"]
)

# Match an opinion
matches = matcher.match_opinion("Scientific evidence supports global warming", top_k=1)
print(matches)
```

### 5. Test Conclusion Generator (Requires OPENAI_API_KEY)
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from models import ConclusionGenerator

generator = ConclusionGenerator()
generator.initialize()

conclusion = generator.generate(
    topic_text="Mars face is a natural landform",
    opinions=[
        {"text": "No life exists on Mars", "type": "Claim"},
        {"text": "Some believe aliens made it", "type": "Counterclaim"}
    ]
)
print(conclusion)
```

### 6. Run FastAPI Server
```bash
python main.py api --port 8000
# Then visit http://localhost:8000/docs
```

### 7. Train Classifier (Full Training)
```bash
python main.py train-classifier \
    --topics-path data/topics.csv \
    --opinions-path data/opinions.csv \
    --conclusions-path data/conclusions.csv \
    --epochs 3
```

### 8. Run Evaluation
```bash
python main.py evaluate
```

---

## What Can Be Tested Right Now

| Feature | Command/Test | Dependencies |
|---------|--------------|--------------|
| Data loading | `python -c "from data import DataProcessor; ..."` | pandas, sklearn |
| Base classifier | `python -c "from models import OpinionClassifier; ..."` | transformers, torch |
| Topic matcher | `python -c "from models import TopicMatcher; ..."` | sentence-transformers |
| FastAPI (no models) | `python main.py api` | fastapi, uvicorn |
| Full training | `python main.py train-classifier` | All ML deps |
| Conclusion gen | Requires `OPENAI_API_KEY` | openai |
| Redis events | Requires running Redis | redis |
| gRPC | After proto compilation | grpcio |

---

## After Creating Proto Files

### Compile Protos
```bash
python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./grpc_service \
    --grpc_python_out=./grpc_service \
    ./protos/opinion_service.proto
```

### Fix Import in Generated File
In `grpc_service/opinion_service_pb2_grpc.py`, change:
```python
# From:
import opinion_service_pb2 as opinion__service__pb2
# To:
from grpc_service import opinion_service_pb2 as opinion__service__pb2
```

### Test gRPC Server
```bash
python main.py grpc --port 50051
```

---

## Key Technical Notes

### Class Imbalance in Data
```
Evidence:     12,105 (44.7%)
Claim:        11,977 (44.2%)
Counterclaim:  1,773 (6.5%)
Rebuttal:      1,244 (4.6%)
```
Use **F1-weighted** as primary metric to handle imbalance.

### Important Constraint
> `topic_id` column is for **VALIDATION ONLY** - do not use as a feature during inference!

### Model Choices
- Topic Matching: `sentence-transformers/all-MiniLM-L6-v2`
- Classification: `distilbert-base-uncased` (fine-tuned)
- Conclusion: `gpt-4o-mini` (configurable)

---

## Environment Variables

```bash
# Required for conclusion generation
export OPENAI_API_KEY="sk-..."

# Optional - defaults shown
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export GRPC_HOST="localhost"
export GRPC_PORT="50051"
```

---

## File Paths to Update

If data files are in different locations, update paths in:
1. `config/settings.py` - `DATA_DIR`
2. CLI commands - `--topics-path`, `--opinions-path`, `--conclusions-path`
3. Test scripts

---

## Priority Tasks for Claude Code

1. **Create proto files** (see content below)
2. **Compile protos** and fix imports
3. **Test basic functionality** with data files
4. **Run classifier training** (even 1 epoch to verify)
5. **Test FastAPI endpoints**
6. **Run evaluation** to get F1 scores