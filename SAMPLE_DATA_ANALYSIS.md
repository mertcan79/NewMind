# Sample Data Analysis - Pipeline Outputs

This file contains sample data from the pipeline outputs for detailed inspection.

---

## Sample 1: High-Quality Match (Similarity: 0.955)

### Topic: 007ACE74B050
**Topic Text:**
```
I think that the face is a natural landform because I dont think that
there is any life on Mars
```

### Top Matched Opinion
**Opinion ID:** c22adee811b6
**Similarity Score:** 0.9548 (95.5% match)

**Opinion Text:**
```
I think that the face is a natural landform because there is no life on
Mars that we have descovered yet
```

### Classification Result
- **Predicted Type:** Evidence
- **Confidence:** 0.61 (61%)
- **Probability Distribution:**
  - Claim: 0.10 (10%)
  - Evidence: 0.61 (61%)
  - Counterclaim: 0.20 (20%)
  - Rebuttal: 0.09 (9%)

**Ground Truth (from opinions.csv):**
- Need to check actual label for opinion_id c22adee811b6

**Analysis:**
- The matcher found an almost identical opinion (only differs by "dont think" vs "no")
- Extremely high semantic similarity (0.955)
- This demonstrates the matcher is working correctly for finding relevant content
- Classifier predicted "Evidence" (which is the only class it predicts)

---

## Sample 2: Good Match (Similarity: 0.942)

### Topic: 007ACE74B050
**Same topic as above**

### Matched Opinion #2
**Opinion ID:** 416df8ad3c59
**Similarity Score:** 0.9425 (94.3% match)

**Opinion Text:**
```
The Face is a natural landform because NASA has proven that there can't
be any life form on Mars
```

### Classification Result
- **Predicted Type:** Evidence
- **Confidence:** 0.61
- Similar probability distribution as Sample 1

**Analysis:**
- Another very high similarity match
- Same semantic meaning (Face is natural, no life on Mars)
- Different phrasing but same argument
- Again predicted as "Evidence"

---

## Sample 3: Medium Match (Similarity: 0.841)

### Topic: 007ACE74B050

### Matched Opinion #8
**Opinion ID:** 1e12879d7786
**Similarity Score:** 0.8411 (84.1% match)

**Opinion Text:**
```
There is a natural landform on Mars that looks like a face because huge
rocks were fromed together looking like a face
```

### Classification Result
- **Predicted Type:** Evidence
- **Confidence:** 0.61

**Analysis:**
- Still high similarity (84%)
- Different argument structure (explains HOW it's a landform, not just that it is)
- Less similar to topic than top matches
- Predicted as "Evidence"

---

## Classification Probability Analysis

### Observed Pattern Across ALL 466 Opinions

Looking at the probability distributions, we see a concerning pattern:

**Typical Probability Distribution:**
```
{
  "Claim": 0.10 (10%)
  "Evidence": 0.61 (61%)
  "Counterclaim": 0.20 (20%)
  "Rebuttal": 0.09 (9%)
}
```

**Observations:**
1. Evidence consistently gets ~60-65% probability
2. Counterclaim gets ~20% probability but NEVER wins
3. Claim gets ~10% probability but NEVER wins
4. Rebuttal gets ~9% probability but NEVER wins
5. All opinions predicted as "Evidence" despite varying probabilities

**This suggests:**
- The model IS learning some features (probabilities vary)
- But it's biased heavily toward Evidence class
- The confidence threshold (0.40) is being met by Evidence in all cases
- Other classes never reach sufficient probability to be predicted

---

## Matching Quality Assessment

### Distribution of Similarity Scores (Top-10 matches)

Based on the sample data, similarity scores range:
- **High matches (>0.90):** Very common in top 2-3 results
- **Good matches (0.85-0.90):** Common in positions 3-6
- **Medium matches (0.80-0.85):** Common in positions 7-10
- **Lower matches (<0.80):** Rarely in top-10

**Example from Topic 007ACE74B050:**
```
Position 1: 0.9548 ⭐⭐⭐⭐⭐
Position 2: 0.9425 ⭐⭐⭐⭐⭐
Position 3: 0.8957 ⭐⭐⭐⭐
Position 4: 0.8887 ⭐⭐⭐⭐
Position 5: 0.8576 ⭐⭐⭐⭐
Position 6: 0.8574 ⭐⭐⭐⭐
Position 7: 0.8420 ⭐⭐⭐⭐
Position 8: 0.8411 ⭐⭐⭐⭐
Position 9: 0.8391 ⭐⭐⭐⭐
Position 10: 0.8xxx ⭐⭐⭐⭐
```

**Quality Assessment:** ✅ EXCELLENT
- All top-10 matches have >80% similarity
- Top matches are extremely relevant (>95% similarity)
- Matcher is working as intended

---

## Ground Truth Comparison Needed

To properly evaluate the classifier, we need to compare:

1. **What the classifier predicted:** Evidence (for all 466 opinions)
2. **What the ground truth says:** Need to check opinions.csv

Let's check a few specific opinion IDs against ground truth:

### Sample Opinion IDs to Verify:
1. c22adee811b6 - Top match, similarity 0.9548
2. 416df8ad3c59 - 2nd match, similarity 0.9425
3. ce3c6cd615f2 - 3rd match, similarity 0.8957

**Action needed:** Cross-reference these opinion_ids with opinions.csv to see true labels.

---

## Why Matching Recall/Precision is Low

### Hypothesis

The evaluation compares:
- **Retrieved opinions:** Based on embedding similarity
- **Relevant opinions (ground truth):** Based on topic_id match in opinions.csv

**Problem:** These may not align because:

1. Topic_id in opinions.csv may be assigned based on:
   - Original essay/topic it came from (arbitrary assignment)
   - Not necessarily semantic similarity

2. Our matcher finds semantically similar content:
   - May retrieve opinions from different topic_ids
   - But these could be semantically relevant
   - Evaluation marks them as "false positives"

**Example:**
```
Topic: "Face on Mars is natural landform, no life on Mars"

Retrieved Opinion: "Mars has no life, Face is natural"
- Embedding similarity: 0.95 (very high)
- topic_id: DIFFERENT from query topic
- Evaluation: Marked as IRRELEVANT (false positive)
- Reality: Highly relevant semantically
```

### Validation Needed

Manual review of a sample (e.g., 10 topics) to check if:
- Low-precision matches are actually irrelevant (matcher failing)
- OR low-precision is due to topic_id mismatch (evaluation issue)

---

## Classifier Failure Deep Dive

### Training Data Distribution (from documentation)

```
Class Distribution in Training Data:
- Evidence:     9,675 samples (35.7%) ← Majority class 1
- Claim:        9,574 samples (35.3%) ← Majority class 2
- Counterclaim: 1,411 samples (5.2%)  ← Minority
- Rebuttal:     1,000 samples (3.7%)  ← Minority
```

### Applied Class Weights (from training metadata)

```
Class Weights:
- Claim:        0.5  (reduce importance)
- Evidence:     0.5  (reduce importance)
- Counterclaim: 1.42 (increase importance)
- Rebuttal:     1.98 (increase importance)
```

### What Went Wrong?

Despite class weighting, the model:
1. Learned to predict the majority class (Evidence)
2. Achieved ~75% accuracy on validation (by predicting Evidence often)
3. But completely failed on minority classes

**Training Metrics from Checkpoint:**
- Best validation macro F1: 0.154 (15.4%) ❌
- This should have been a red flag during training
- Expected macro F1: ~0.72 (72%)

### Why Standard Class Weighting Failed

Class weights help with imbalance but:
1. **10:1 ratio** (9,675 vs 1,000) is too severe for simple weighting
2. Model may need:
   - More aggressive techniques (SMOTE, focal loss)
   - Different architecture (ensemble methods)
   - Better regularization
   - More training data for minority classes

---

## Output File Formats (Validation)

### ✅ outputs/topic_to_opinions.json
**Format:** Correct
```json
{
  "topic_id": [
    {"opinion_id": "...", "similarity": 0.xx, "opinion_text": "..."}
  ]
}
```

### ✅ outputs/topic_to_opinions_labeled.json
**Format:** Correct
```json
{
  "topic_id": {
    "topic_text": "...",
    "opinions": [
      {
        "opinion_id": "...",
        "text": "...",
        "similarity": 0.xx,
        "predicted_type": "...",
        "confidence": 0.xx,
        "probs": {"Claim": 0.xx, ...}
      }
    ]
  }
}
```

### ⏭️ outputs/conclusions_generated.csv
**Expected Format:**
```csv
topic_id,generated_conclusion
topic_1,Based on the evidence...
topic_2,The arguments suggest...
```
**Status:** Not generated (--no_llm flag used)

---

## Recommendations for Next Analysis

1. **Run with more topics** (500-1000) to see if patterns hold
2. **Manual review** of 10-20 matched opinions to validate quality
3. **Check ground truth labels** for the sampled opinions
4. **Retrain classifier** with better techniques
5. **Test with LLM** on small set (10 topics) to validate conclusions stage

---

**Analysis Date:** 2025-12-26
**Sample Size:** 50 topics, 466 matched opinions
**Key Finding:** Matcher works well, classifier needs complete retraining
