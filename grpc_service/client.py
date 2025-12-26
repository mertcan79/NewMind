"""
gRPC Client for Opinion Analysis Service.

A client for testing and interacting with the gRPC server.

Usage:
    from grpc_service import OpinionAnalyzerClient
    
    client = OpinionAnalyzerClient()
    client.connect()
    
    # Classify an opinion
    result = client.classify_opinion("I think this is true")
    print(result)
    
    client.close()
"""
import grpc
from pathlib import Path
import sys
from typing import List, Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

# Lazy import of protobuf modules
pb2 = None
pb2_grpc = None


def _load_protos():
    """Lazily load protobuf modules."""
    global pb2, pb2_grpc
    if pb2 is None:
        try:
            from grpc_service import opinion_service_pb2, opinion_service_pb2_grpc
            pb2 = opinion_service_pb2
            pb2_grpc = opinion_service_pb2_grpc
            return True
        except ImportError as e:
            print(f"Error loading protos: {e}")
            print("Please compile protos first. See grpc_service/__init__.py for instructions.")
            return False
    return True


class OpinionAnalyzerClient:
    """
    gRPC client for the Opinion Analysis service.
    
    Provides easy-to-use methods for all service endpoints.
    
    Example:
        client = OpinionAnalyzerClient(host="localhost", port=50051)
        client.connect()
        
        result = client.classify_opinion("Scientific evidence supports this")
        print(result)
        # {'success': True, 'predicted_type': 'Evidence', ...}
        
        client.close()
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize client.
        
        Args:
            host: Server host (default: from settings)
            port: Server port (default: from settings)
        """
        self.host = host or settings.GRPC_HOST
        self.port = port or settings.GRPC_PORT
        self.channel = None
        self.stub = None
        
    def connect(self, timeout: int = 10):
        """
        Establish connection to the gRPC server.
        
        Args:
            timeout: Connection timeout in seconds
        """
        if not _load_protos():
            raise RuntimeError("Cannot connect: protos not compiled")
        
        address = f"{self.host}:{self.port}"
        
        self.channel = grpc.insecure_channel(address)
        
        # Wait for channel to be ready
        try:
            grpc.channel_ready_future(self.channel).result(timeout=timeout)
        except grpc.FutureTimeoutError:
            raise ConnectionError(f"Could not connect to gRPC server at {address}")
        
        self.stub = pb2_grpc.OpinionAnalyzerStub(self.channel)
        print(f"Connected to gRPC server at {address}")
        
    def close(self):
        """Close the connection."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
            print("Connection closed")
    
    def _ensure_connected(self):
        """Ensure client is connected."""
        if self.stub is None:
            raise RuntimeError("Not connected. Call connect() first.")
    
    def health_check(self) -> Dict:
        """
        Check service health.
        
        Returns:
            Dict with health status and service availability
        """
        self._ensure_connected()
        
        response = self.stub.HealthCheck(pb2.HealthRequest())
        
        return {
            "healthy": response.healthy,
            "status": response.status,
            "services": dict(response.services)
        }
    
    def match_opinion_to_topic(
        self,
        opinion_text: str,
        top_k: int = 1,
        threshold: float = 0.5
    ) -> Dict:
        """
        Match an opinion to topics.
        
        Args:
            opinion_text: The opinion text to match
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            Dict with matches and success status
        """
        self._ensure_connected()
        
        request = pb2.MatchRequest(
            opinion_text=opinion_text,
            top_k=top_k,
            threshold=threshold
        )
        
        response = self.stub.MatchOpinionToTopic(request)
        
        return {
            "success": response.success,
            "error_message": response.error_message,
            "matches": [
                {
                    "topic_id": m.topic_id,
                    "topic_text": m.topic_text,
                    "similarity": m.similarity
                }
                for m in response.matches
            ]
        }
    
    def classify_opinion(self, opinion_text: str) -> Dict:
        """
        Classify an opinion type.
        
        Args:
            opinion_text: The opinion to classify
            
        Returns:
            Dict with predicted type, id, probabilities
        """
        self._ensure_connected()
        
        request = pb2.ClassifyRequest(opinion_text=opinion_text)
        response = self.stub.ClassifyOpinion(request)
        
        return {
            "success": response.success,
            "error_message": response.error_message,
            "predicted_type": response.predicted_type,
            "predicted_id": response.predicted_id,
            "probabilities": dict(response.probabilities)
        }
    
    def classify_opinions_batch(self, opinion_texts: List[str]) -> Dict:
        """
        Classify multiple opinions in batch.
        
        Args:
            opinion_texts: List of opinions to classify
            
        Returns:
            Dict with list of classification results
        """
        self._ensure_connected()
        
        request = pb2.BatchClassifyRequest(opinion_texts=opinion_texts)
        response = self.stub.ClassifyOpinionsBatch(request)
        
        return {
            "success": response.success,
            "error_message": response.error_message,
            "results": [
                {
                    "predicted_type": r.predicted_type,
                    "predicted_id": r.predicted_id,
                    "probabilities": dict(r.probabilities)
                }
                for r in response.results
            ]
        }
    
    def generate_conclusion(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200
    ) -> Dict:
        """
        Generate a conclusion from topic and opinions.
        
        Args:
            topic_text: The main topic text
            opinions: List of dicts with 'text' and 'type' keys
            temperature: OpenAI temperature
            max_tokens: Max tokens in response
            
        Returns:
            Dict with generated conclusion
        """
        self._ensure_connected()
        
        opinion_messages = [
            pb2.Opinion(text=op["text"], type=op["type"])
            for op in opinions
        ]
        
        request = pb2.ConclusionRequest(
            topic_text=topic_text,
            opinions=opinion_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = self.stub.GenerateConclusion(request)
        
        return {
            "success": response.success,
            "error_message": response.error_message,
            "conclusion": response.conclusion
        }
    
    def analyze_opinions(
        self,
        topic_text: str,
        opinion_texts: List[str],
        generate_conclusion: bool = True
    ) -> Dict:
        """
        Run full analysis pipeline.
        
        Args:
            topic_text: The main topic
            opinion_texts: List of opinion texts
            generate_conclusion: Whether to generate conclusion
            
        Returns:
            Dict with classified opinions and optional conclusion
        """
        self._ensure_connected()
        
        request = pb2.AnalyzeRequest(
            topic_text=topic_text,
            opinion_texts=opinion_texts,
            generate_conclusion=generate_conclusion
        )
        
        response = self.stub.AnalyzeOpinions(request)
        
        return {
            "success": response.success,
            "error_message": response.error_message,
            "topic_text": response.topic_text,
            "classified_opinions": [
                {
                    "text": co.text,
                    "predicted_type": co.predicted_type,
                    "confidence": co.confidence
                }
                for co in response.classified_opinions
            ],
            "conclusion": response.conclusion
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
    print("="*50)
    print("Testing gRPC Client")
    print("="*50)
    
    try:
        with OpinionAnalyzerClient() as client:
            # Health check
            print("\n1. Health Check:")
            health = client.health_check()
            print(f"   Healthy: {health['healthy']}")
            print(f"   Status: {health['status']}")
            print(f"   Services: {health['services']}")
            
            # Classify opinion
            print("\n2. Classify Opinion:")
            result = client.classify_opinion(
                "I think climate change is real because of scientific evidence"
            )
            if result['success']:
                print(f"   Type: {result['predicted_type']}")
                print(f"   Probabilities: {result['probabilities']}")
            else:
                print(f"   Error: {result['error_message']}")
            
            # Full analysis
            print("\n3. Full Analysis:")
            result = client.analyze_opinions(
                topic_text="Climate change is a serious threat",
                opinion_texts=[
                    "Scientific consensus supports this view",
                    "Some argue the data is inconclusive"
                ],
                generate_conclusion=True
            )
            if result['success']:
                for co in result['classified_opinions']:
                    print(f"   - {co['predicted_type']}: {co['text'][:50]}...")
                if result['conclusion']:
                    print(f"   Conclusion: {result['conclusion'][:100]}...")
            else:
                print(f"   Error: {result['error_message']}")
                
    except ConnectionError as e:
        print(f"\nConnection error: {e}")
        print("Make sure the gRPC server is running:")
        print("  python main.py grpc")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    test_client()