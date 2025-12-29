"""
gRPC Client for Opinion Analysis Service.

A client for testing and interacting with the gRPC server.

Usage:
    from grpc_service.client import OpinionAnalysisClient

    client = OpinionAnalysisClient()
    client.connect()

    # Classify an opinion
    result = client.classify_opinion("I think this is true")
    print(result)

    client.close()
"""
import sys
from pathlib import Path
from typing import Dict, List

import grpc

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from grpc_service import opinion_service_pb2, opinion_service_pb2_grpc


class OpinionAnalysisClient:
    """
    gRPC client for the Opinion Analysis service.

    Provides easy-to-use methods for all service endpoints.

    Example:
        client = OpinionAnalysisClient(host="localhost", port=50051)
        client.connect()

        result = client.classify_opinion("Scientific evidence supports this")
        print(result)
        # {'opinion_type': 'Evidence', 'confidence': 0.95, 'probabilities': {...}}

        client.close()
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize client.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None

    def connect(self, timeout: int = 10):
        """
        Establish connection to the gRPC server.

        Args:
            timeout: Connection timeout in seconds
        """
        address = f"{self.host}:{self.port}"

        self.channel = grpc.insecure_channel(address)

        # Wait for channel to be ready
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"Could not connect to gRPC server at {address}")

        self.stub = opinion_service_pb2_grpc.OpinionAnalysisServiceStub(self.channel)
        print(f"✓ Connected to gRPC server at {address}")

    def close(self):
        """Close the connection."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
            print("✓ Connection closed")

    def _ensure_connected(self):
        """Ensure client is connected."""
        if self.stub is None:
            raise RuntimeError("Not connected. Call connect() first.")

    def classify_opinion(self, text: str) -> Dict:
        """
        Classify a single opinion.

        Args:
            text: The opinion text to classify

        Returns:
            Dict with opinion_type, confidence, and probabilities
        """
        self._ensure_connected()

        request = opinion_service_pb2.ClassifyRequest(text=text)
        response = self.stub.ClassifyOpinion(request)

        return {
            "opinion_type": response.opinion_type,
            "confidence": response.confidence,
            "probabilities": dict(response.probabilities)
        }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """
        Classify multiple opinions in batch.

        Args:
            texts: List of opinion texts to classify

        Returns:
            List of dicts with opinion_type, confidence, probabilities
        """
        self._ensure_connected()

        request = opinion_service_pb2.BatchClassifyRequest(texts=texts)
        response = self.stub.ClassifyBatch(request)

        return [
            {
                "opinion_type": r.opinion_type,
                "confidence": r.confidence,
                "probabilities": dict(r.probabilities)
            }
            for r in response.results
        ]

    def match_opinions_to_topic(
        self,
        topic_id: str,
        topic_text: str,
        opinions: List[Dict[str, str]],
        top_k: int = 10,
        threshold: float = None
    ) -> List[Dict]:
        """
        Match opinions to a topic.

        Args:
            topic_id: Topic identifier
            topic_text: Topic text
            opinions: List of dicts with 'id' and 'text' keys
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold (optional)

        Returns:
            List of matched opinions with similarity scores
        """
        self._ensure_connected()

        opinion_messages = [
            opinion_service_pb2.Opinion(id=op["id"], text=op["text"])
            for op in opinions
        ]

        request = opinion_service_pb2.MatchRequest(
            topic_id=topic_id,
            topic_text=topic_text,
            opinions=opinion_messages,
            top_k=top_k
        )

        if threshold is not None:
            request.threshold = threshold

        response = self.stub.MatchOpinionsToTopic(request)

        return [
            {
                "opinion_id": m.opinion_id,
                "opinion_text": m.opinion_text,
                "similarity": m.similarity
            }
            for m in response.matches
        ]

    def generate_conclusion(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]]
    ) -> str:
        """
        Generate a conclusion for a topic with classified opinions.

        Args:
            topic_text: The main topic text
            opinions: List of dicts with 'text' and 'opinion_type' keys

        Returns:
            Generated conclusion text
        """
        self._ensure_connected()

        classified_opinions = [
            opinion_service_pb2.ClassifiedOpinion(
                text=op["text"],
                opinion_type=op["opinion_type"]
            )
            for op in opinions
        ]

        request = opinion_service_pb2.ConclusionRequest(
            topic_text=topic_text,
            opinions=classified_opinions
        )

        response = self.stub.GenerateConclusion(request)
        return response.conclusion

    def analyze_topic(
        self,
        topic_id: str,
        topic_text: str,
        all_opinions: List[Dict[str, str]],
        top_k: int = 10,
        generate_conclusion: bool = True
    ) -> Dict:
        """
        Run full analysis pipeline: match + classify + conclude.

        Args:
            topic_id: Topic identifier
            topic_text: Topic text
            all_opinions: List of all opinions with 'id' and 'text' keys
            top_k: Number of opinions to match
            generate_conclusion: Whether to generate conclusion

        Returns:
            Dict with matched_opinions, classified_opinions, conclusion, metrics
        """
        self._ensure_connected()

        opinion_messages = [
            opinion_service_pb2.Opinion(id=op["id"], text=op["text"])
            for op in all_opinions
        ]

        request = opinion_service_pb2.TopicAnalysisRequest(
            topic_id=topic_id,
            topic_text=topic_text,
            all_opinions=opinion_messages,
            top_k=top_k,
            generate_conclusion=generate_conclusion
        )

        response = self.stub.AnalyzeTopic(request)

        return {
            "topic_id": response.topic_id,
            "matched_opinions": [
                {
                    "opinion_id": m.opinion_id,
                    "opinion_text": m.opinion_text,
                    "similarity": m.similarity
                }
                for m in response.matched_opinions
            ],
            "classified_opinions": [
                {
                    "text": co.text,
                    "opinion_type": co.opinion_type
                }
                for co in response.classified_opinions
            ],
            "conclusion": response.conclusion if response.HasField("conclusion") else None,
            "metrics": {
                "total_opinions_analyzed": response.metrics.total_opinions_analyzed,
                "matched_opinions": response.metrics.matched_opinions,
                "avg_similarity": response.metrics.avg_similarity,
                "type_distribution": dict(response.metrics.type_distribution)
            }
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def test_client():
    """Test client functionality."""
    print("="*60)
    print("Testing gRPC Opinion Analysis Client")
    print("="*60)

    try:
        with OpinionAnalysisClient() as client:
            # Test 1: Classify single opinion
            print("\n1. Classify Single Opinion:")
            result = client.classify_opinion(
                "I believe climate change is real based on scientific evidence"
            )
            print(f"   Type: {result['opinion_type']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Probabilities: {result['probabilities']}")

            # Test 2: Classify batch
            print("\n2. Classify Batch:")
            results = client.classify_batch([
                "Scientific data supports this conclusion",
                "However, some argue that the data is inconclusive",
                "This undermines the previous argument"
            ])
            for i, r in enumerate(results, 1):
                print(f"   Opinion {i}: {r['opinion_type']} (confidence: {r['confidence']:.3f})")

            # Test 3: Full topic analysis
            print("\n3. Full Topic Analysis:")
            result = client.analyze_topic(
                topic_id="topic_001",
                topic_text="Climate change is a serious global threat",
                all_opinions=[
                    {"id": "op1", "text": "Scientific consensus supports this view"},
                    {"id": "op2", "text": "Temperatures have risen significantly"},
                    {"id": "op3", "text": "Some question the reliability of climate models"},
                    {"id": "op4", "text": "However, the evidence is overwhelming"},
                    {"id": "op5", "text": "We must take action now"}
                ],
                top_k=5,
                generate_conclusion=True
            )
            print(f"   Matched: {result['metrics']['matched_opinions']} opinions")
            print(f"   Avg similarity: {result['metrics']['avg_similarity']:.3f}")
            print(f"   Type distribution: {result['metrics']['type_distribution']}")
            if result['conclusion']:
                print(f"   Conclusion: {result['conclusion'][:100]}...")

            print("\n✓ All tests passed!")

    except ConnectionError as e:
        print(f"\n✗ Connection error: {e}")
        print("\nMake sure the gRPC server is running:")
        print("  python grpc_service/server.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_client()
