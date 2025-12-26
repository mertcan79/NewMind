"""
gRPC Server for Opinion Analysis Service.

This server exposes ML models via gRPC for high-performance inference.

Usage:
    # Start server with models
    from grpc_service import serve
    from models import TopicMatcher, OpinionClassifier, ConclusionGenerator
    
    classifier = OpinionClassifier()
    classifier.load_model()
    
    serve(classifier=classifier, port=50051)
"""
import grpc
from concurrent import futures
from pathlib import Path
import sys
import time
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

# Lazy import of protobuf modules (may not be compiled yet)
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


class OpinionAnalyzerServicer:
    """
    gRPC service implementation for opinion analysis.
    
    Provides endpoints for:
    - Topic matching (sentence similarity)
    - Opinion classification (DistilBERT)
    - Conclusion generation (OpenAI)
    - Full analysis pipeline
    """
    
    def __init__(
        self,
        topic_matcher=None,
        classifier=None,
        conclusion_generator=None
    ):
        """
        Initialize servicer with ML models.
        
        Args:
            topic_matcher: Initialized TopicMatcher instance
            classifier: Initialized OpinionClassifier instance
            conclusion_generator: Initialized ConclusionGenerator instance
        """
        self.topic_matcher = topic_matcher
        self.classifier = classifier
        self.conclusion_generator = conclusion_generator
        
        self._services = {
            "topic_matcher": topic_matcher is not None,
            "classifier": classifier is not None,
            "conclusion_generator": conclusion_generator is not None
        }
        
        print("OpinionAnalyzerServicer initialized:")
        for service, status in self._services.items():
            print(f"  {service}: {'✓' if status else '✗'}")
    
    def MatchOpinionToTopic(self, request, context):
        """Match an opinion to the most relevant topic(s)."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Protos not compiled")
            return pb2.MatchResponse() if pb2 else None
        
        try:
            if not self._services["topic_matcher"]:
                return pb2.MatchResponse(
                    success=False,
                    error_message="Topic matcher not initialized"
                )
            
            top_k = request.top_k if request.top_k > 0 else 1
            threshold = request.threshold if request.threshold > 0 else settings.SIMILARITY_THRESHOLD
            
            matches = self.topic_matcher.match_opinion(
                opinion_text=request.opinion_text,
                top_k=top_k,
                threshold=threshold
            )
            
            topic_matches = [
                pb2.TopicMatch(
                    topic_id=m["topic_id"],
                    topic_text=m["topic_text"],
                    similarity=m["similarity"]
                )
                for m in matches
            ]
            
            return pb2.MatchResponse(matches=topic_matches, success=True)
            
        except Exception as e:
            return pb2.MatchResponse(success=False, error_message=str(e))
    
    def ClassifyOpinion(self, request, context):
        """Classify a single opinion."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ClassifyResponse() if pb2 else None
        
        try:
            if not self._services["classifier"]:
                return pb2.ClassifyResponse(
                    success=False,
                    error_message="Classifier not initialized"
                )
            
            result = self.classifier.predict_single(request.opinion_text)
            
            return pb2.ClassifyResponse(
                predicted_type=result["predicted_type"],
                predicted_id=result["predicted_id"],
                probabilities=result["probabilities"],
                success=True
            )
            
        except Exception as e:
            return pb2.ClassifyResponse(success=False, error_message=str(e))
    
    def ClassifyOpinionsBatch(self, request, context):
        """Classify multiple opinions in batch."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.BatchClassifyResponse() if pb2 else None
        
        try:
            if not self._services["classifier"]:
                return pb2.BatchClassifyResponse(
                    success=False,
                    error_message="Classifier not initialized"
                )
            
            results = []
            for text in request.opinion_texts:
                result = self.classifier.predict_single(text)
                results.append(pb2.ClassifyResponse(
                    predicted_type=result["predicted_type"],
                    predicted_id=result["predicted_id"],
                    probabilities=result["probabilities"],
                    success=True
                ))
            
            return pb2.BatchClassifyResponse(results=results, success=True)
            
        except Exception as e:
            return pb2.BatchClassifyResponse(success=False, error_message=str(e))
    
    def GenerateConclusion(self, request, context):
        """Generate conclusion from topic and opinions."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ConclusionResponse() if pb2 else None
        
        try:
            if not self._services["conclusion_generator"]:
                return pb2.ConclusionResponse(
                    success=False,
                    error_message="Conclusion generator not initialized. Set OPENAI_API_KEY."
                )
            
            opinions = [
                {"text": op.text, "type": op.type}
                for op in request.opinions
            ]
            
            temperature = request.temperature if request.temperature > 0 else 0.7
            max_tokens = request.max_tokens if request.max_tokens > 0 else 200
            
            conclusion = self.conclusion_generator.generate(
                topic_text=request.topic_text,
                opinions=opinions,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return pb2.ConclusionResponse(conclusion=conclusion, success=True)
            
        except Exception as e:
            return pb2.ConclusionResponse(success=False, error_message=str(e))
    
    def AnalyzeOpinions(self, request, context):
        """Full pipeline: classify opinions and optionally generate conclusion."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.AnalyzeResponse() if pb2 else None
        
        try:
            classified_opinions = []
            
            # Step 1: Classify all opinions
            if self._services["classifier"]:
                for opinion_text in request.opinion_texts:
                    result = self.classifier.predict_single(opinion_text)
                    classified_opinions.append(
                        pb2.ClassifiedOpinion(
                            text=opinion_text,
                            predicted_type=result["predicted_type"],
                            confidence=max(result["probabilities"].values())
                        )
                    )
            
            # Step 2: Generate conclusion if requested
            conclusion = ""
            if request.generate_conclusion and self._services["conclusion_generator"]:
                opinions = [
                    {"text": co.text, "type": co.predicted_type}
                    for co in classified_opinions
                ]
                
                conclusion = self.conclusion_generator.generate(
                    topic_text=request.topic_text,
                    opinions=opinions
                )
            
            return pb2.AnalyzeResponse(
                topic_text=request.topic_text,
                classified_opinions=classified_opinions,
                conclusion=conclusion,
                success=True
            )
            
        except Exception as e:
            return pb2.AnalyzeResponse(success=False, error_message=str(e))
    
    def HealthCheck(self, request, context):
        """Health check endpoint."""
        
        if not _load_protos():
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.HealthResponse() if pb2 else None
        
        all_healthy = all(self._services.values())
        
        return pb2.HealthResponse(
            healthy=all_healthy,
            status="All services running" if all_healthy else "Some services not initialized",
            services=self._services
        )


def serve(
    topic_matcher=None,
    classifier=None,
    conclusion_generator=None,
    host: str = None,
    port: int = None,
    max_workers: int = 10
):
    """
    Start the gRPC server.
    
    Args:
        topic_matcher: Initialized TopicMatcher (optional)
        classifier: Initialized OpinionClassifier (optional)
        conclusion_generator: Initialized ConclusionGenerator (optional)
        host: Server host (default: from settings)
        port: Server port (default: from settings)
        max_workers: Thread pool size
    
    Example:
        from models import OpinionClassifier
        from grpc_service import serve
        
        classifier = OpinionClassifier()
        classifier.load_model()
        
        serve(classifier=classifier)
    """
    if not _load_protos():
        print("Cannot start server: protos not compiled")
        return
    
    host = host or settings.GRPC_HOST
    port = port or settings.GRPC_PORT
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    servicer = OpinionAnalyzerServicer(
        topic_matcher=topic_matcher,
        classifier=classifier,
        conclusion_generator=conclusion_generator
    )
    
    pb2_grpc.add_OpinionAnalyzerServicer_to_server(servicer, server)
    
    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)
    
    print(f"\n{'='*50}")
    print(f"Starting gRPC server on {server_address}")
    print(f"{'='*50}\n")
    
    server.start()
    
    try:
        while True:
            time.sleep(86400)  # One day
    except KeyboardInterrupt:
        print("\nShutting down gRPC server...")
        server.stop(grace=5)
        print("Server stopped.")


if __name__ == "__main__":
    # Test server startup (no models loaded)
    print("Starting gRPC server in test mode (no models)...")
    serve()