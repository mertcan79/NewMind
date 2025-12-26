"""
CONCLUSION GENERATOR MODULE
Uses OpenAI API to generate conclusions from grouped opinions.
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class ConclusionGenerator:
    """
    GENERATES CONCLUSION SUMMARIES FROM TOPICS AND THEIR OPINIONS USING OPENAI
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.OPENAI_MODEL
        self.client = None

    def initialize(self):
        """INITIALIZE THE OPENAI CLIENT"""
        from openai import OpenAI

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        print(f"Initialized OpenAI client with model: {self.model}")

    def _build_prompt(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]]
    ) -> str:
        """
        BUILD THE PROMPT FOR CONCLUSION GENERATION

        Args:
            topic_text: The main topic/position text
            opinions: List of dicts with 'text' and 'type' keys
        """

        # GROUP OPINIONS BY TYPE
        claims = [o["text"] for o in opinions if o["type"] == "Claim"]
        counterclaims = [o["text"] for o in opinions if o["type"] == "Counterclaim"]
        rebuttals = [o["text"] for o in opinions if o["type"] == "Rebuttal"]
        evidence = [o["text"] for o in opinions if o["type"] == "Evidence"]

        prompt = f"""You are a social media analyst. Given a main topic/position and related opinions from social media, write a concise conclusion that summarizes the overall sentiment and key points.

MAIN TOPIC/POSITION:
{topic_text}

SUPPORTING CLAIMS:
{self._format_list(claims) if claims else "None provided"}

COUNTER CLAIMS (opposing views):
{self._format_list(counterclaims) if counterclaims else "None provided"}

REBUTTALS (responses to counter claims):
{self._format_list(rebuttals) if rebuttals else "None provided"}

EVIDENCE:
{self._format_list(evidence) if evidence else "None provided"}

Based on the above, write a concluding statement that:
1. Summarizes the main position
2. Acknowledges any counter-arguments if present
3. States the overall direction of public opinion on this topic
4. Is 2-4 sentences long

CONCLUSION:"""

        return prompt

    def _format_list(self, items: List[str], max_items: int = 5) -> str:
        """FORMAT A LIST OF ITEMS FOR THE PROMPT"""
        if not items:
            return "None"

        # LIMIT NUMBER OF ITEMS TO AVOID TOKEN LIMITS
        items = items[:max_items]
        return "\n".join(f"- {item.strip()}" for item in items)

    def generate(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> str:
        """
        GENERATE A CONCLUSION FOR A TOPIC AND ITS OPINIONS

        Args:
            topic_text: The main topic/position text
            opinions: List of dicts with 'text' and 'type' keys
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response

        Returns:
            Generated conclusion text
        """
        if self.client is None:
            self.initialize()

        prompt = self._build_prompt(topic_text, opinions)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional social media analyst who summarizes public opinion on various topics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        conclusion = response.choices[0].message.content.strip()
        return conclusion

    def generate_batch(
        self,
        topics: List[str],
        opinions_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> List[str]:
        """
        GENERATE CONCLUSIONS FOR MULTIPLE TOPICS

        Args:
            topics: List of topic texts
            opinions_list: List of opinion lists (one per topic)

        Returns:
            List of generated conclusions
        """
        conclusions = []

        for topic, opinions in zip(topics, opinions_list):
            try:
                conclusion = self.generate(
                    topic_text=topic,
                    opinions=opinions,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                conclusions.append(conclusion)
            except Exception as e:
                print(f"Error generating conclusion: {e}")
                conclusions.append("")

        return conclusions


def evaluate_conclusions(
    generated: List[str],
    references: List[str],
    use_bertscore: bool = True
) -> Dict:
    """
    EVALUATE GENERATED CONCLUSIONS AGAINST REFERENCE CONCLUSIONS

    Uses ROUGE scores and optionally BERTScore.

    Args:
        generated: List of generated conclusions
        references: List of reference conclusions
        use_bertscore: Whether to compute BERTScore (slower but more semantic)

    Returns:
        Dict with evaluation metrics
    """
    from rouge_score import rouge_scorer

    # INITIALIZE ROUGE SCORER
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # COMPUTE ROUGE SCORES
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for gen, ref in zip(generated, references):
        scores = scorer.score(ref, gen)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    results = {
        "rouge1_f1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
        "rouge2_f1": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]),
        "rougeL_f1": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]),
        "num_samples": len(generated)
    }

    # OPTIONALLY COMPUTE BERTSCORE
    if use_bertscore:
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(generated, references, lang="en", verbose=False)
            results["bertscore_precision"] = P.mean().item()
            results["bertscore_recall"] = R.mean().item()
            results["bertscore_f1"] = F1.mean().item()
        except ImportError:
            print("BERTScore not available. Install with: pip install bert-score")

    return results


if __name__ == "__main__":
    # TEST THE GENERATOR (REQUIRES API KEY)
    import os

    # CHECK FOR API KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping live test.")
        print("\nTo test, set the environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
    else:
        # INITIALIZE GENERATOR
        generator = ConclusionGenerator(api_key=api_key)
        generator.initialize()

        # TEST WITH SAMPLE DATA
        topic = "I think that the face is a natural landform because I dont think that there is any life on Mars."

        opinions = [
            {"text": "I think that the face is a natural landform because there is no life on Mars that we have discovered yet", "type": "Claim"},
            {"text": "If life was on Mars, we would know by now. The reason why I think it is a natural landform because, nobody lives on Mars in order to create the figure.", "type": "Evidence"},
            {"text": "People thought that the face was formed by aliens because they thought that there was life on Mars.", "type": "Counterclaim"},
            {"text": "Though some say that life on Mars does exist, I think that there is no life on Mars.", "type": "Rebuttal"}
        ]

        print("Generating conclusion...")
        conclusion = generator.generate(topic, opinions)

        print("\n" + "=" * 60)
        print("GENERATED CONCLUSION:")
        print("=" * 60)
        print(conclusion)
