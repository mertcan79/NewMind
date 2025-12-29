"""
gRPC Server for Opinion Analysis Service.

Implements the OpinionAnalysisService defined in opinion_service.proto.

Usage:
    python grpc_service/server.py
"""
import os
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from grpc_service import opinion_service_pb2, opinion_service_pb2_grpc
from models.conclusion_generator import ConclusionGenerator
from models.opinion_classifier import OpinionClassifier
from models.topic_matcher import TopicMatcher

# Load environment variables
load_dotenv()


class OpinionAnalysisServicer(opinion_service_pb2_grpc.OpinionAnalysisServiceServicer):
    """
    gRPC service implementation for opinion analysis.

    Provides endpoints for:
    - Opinion classification (single and batch)
    - Topic matching
    - Conclusion generation
    - Full topic analysis pipeline
    """

    def __init__(self):
        """Initialize servicer with ML models."""
        print("Initializing OpinionAnalysisService...")

        # Initialize classifier
        try:
            self.classifier = OpinionClassifier()
            print("✓ Classifier loaded")
        except Exception as e:
            print(f"✗ Classifier failed: {e}")
            self.classifier = None

        # Initialize topic matcher
        try:
            self.topic_matcher = TopicMatcher()
            self.topic_matcher.load_model()
            print("✓ Topic matcher loaded")
        except Exception as e:
            print(f"✗ Topic matcher failed: {e}")
            self.topic_matcher = None

        # Initialize conclusion generator
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.conclusion_generator = ConclusionGenerator(api_key=api_key)
                self.conclusion_generator.initialize()
                print("✓ Conclusion generator loaded")
            else:
                print("✗ OPENAI_API_KEY not set - conclusion generation disabled")
                self.conclusion_generator = None
        except Exception as e:
            print(f"✗ Conclusion generator failed: {e}")
            self.conclusion_generator = None

    def ClassifyOpinion(self, request, context):
        """Classify a single opinion."""
        try:
            if not self.classifier:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Classifier not available")
                return opinion_service_pb2.ClassifyResponse()

            result = self.classifier.predict(request.text)

            return opinion_service_pb2.ClassifyResponse(
                opinion_type=result["label"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Classification error: {str(e)}")
            return opinion_service_pb2.ClassifyResponse()

    def ClassifyBatch(self, request, context):
        """Classify multiple opinions in batch."""
        try:
            if not self.classifier:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Classifier not available")
                return opinion_service_pb2.BatchClassifyResponse()

            batch_results = self.classifier.batch_predict(list(request.texts))

            results = []
            for result in batch_results:
                results.append(
                    opinion_service_pb2.ClassifyResponse(
                        opinion_type=result["label"],
                        confidence=result["confidence"],
                        probabilities=result["probabilities"]
                    )
                )

            return opinion_service_pb2.BatchClassifyResponse(results=results)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch classification error: {str(e)}")
            return opinion_service_pb2.BatchClassifyResponse()

    def MatchOpinionsToTopic(self, request, context):
        """Match opinions to a topic."""
        try:
            if not self.topic_matcher:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Topic matcher not available")
                return opinion_service_pb2.MatchResponse()

            # Encode topic
            topic_ids = [request.topic_id]
            topic_texts = [request.topic_text]
            self.topic_matcher.encode_topics(topic_ids, topic_texts)

            # Encode opinions
            opinion_ids = [op.id for op in request.opinions]
            opinion_texts = [op.text for op in request.opinions]
            self.topic_matcher.encode_opinions(opinion_ids, opinion_texts)

            # Compute similarity
            self.topic_matcher.compute_similarity_matrix()

            # Get top matches
            top_k = request.top_k if request.top_k > 0 else 10
            threshold = request.threshold if request.HasField('threshold') else None

            matches = self.topic_matcher.top_opinions_for_topic(
                topic_id=request.topic_id,
                top_k=top_k,
                threshold=threshold
            )

            # Convert to proto format
            matched_opinions = []
            for match in matches:
                matched_opinions.append(
                    opinion_service_pb2.MatchedOpinion(
                        opinion_id=match["opinion_id"],
                        opinion_text=match["opinion_text"],
                        similarity=match["similarity"]
                    )
                )

            return opinion_service_pb2.MatchResponse(matches=matched_opinions)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Matching error: {str(e)}")
            return opinion_service_pb2.MatchResponse()

    def GenerateConclusion(self, request, context):
        """Generate conclusion for a topic with opinions."""
        try:
            if not self.conclusion_generator:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Conclusion generator not available (OPENAI_API_KEY not set)")
                return opinion_service_pb2.ConclusionResponse()

            # Convert proto opinions to dict format
            opinions = [
                {"text": op.text, "type": op.opinion_type}
                for op in request.opinions
            ]

            # Generate conclusion
            conclusion = self.conclusion_generator.generate(
                topic_text=request.topic_text,
                opinions=opinions
            )

            return opinion_service_pb2.ConclusionResponse(conclusion=conclusion)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Conclusion generation error: {str(e)}")
            return opinion_service_pb2.ConclusionResponse()

    def AnalyzeTopic(self, request, context):
        """Full pipeline: match + classify + conclude."""
        try:
            # Stage 1: Match opinions to topic
            matched_opinions = []
            if self.topic_matcher:
                # Encode topic
                self.topic_matcher.encode_topics([request.topic_id], [request.topic_text])

                # Encode all opinions
                opinion_ids = [op.id for op in request.all_opinions]
                opinion_texts = [op.text for op in request.all_opinions]
                self.topic_matcher.encode_opinions(opinion_ids, opinion_texts)

                # Compute similarity and match
                self.topic_matcher.compute_similarity_matrix()
                top_k = request.top_k if request.top_k > 0 else 10

                matches = self.topic_matcher.top_opinions_for_topic(
                    topic_id=request.topic_id,
                    top_k=top_k
                )

                for match in matches:
                    matched_opinions.append(
                        opinion_service_pb2.MatchedOpinion(
                            opinion_id=match["opinion_id"],
                            opinion_text=match["opinion_text"],
                            similarity=match["similarity"]
                        )
                    )

            # Stage 2: Classify matched opinions
            classified_opinions = []
            type_distribution = {}
            total_similarity = 0.0

            if self.classifier and matched_opinions:
                for matched_op in matched_opinions:
                    result = self.classifier.predict(matched_op.opinion_text)
                    opinion_type = result["label"]

                    classified_opinions.append(
                        opinion_service_pb2.ClassifiedOpinion(
                            text=matched_op.opinion_text,
                            opinion_type=opinion_type
                        )
                    )

                    # Update type distribution
                    type_distribution[opinion_type] = type_distribution.get(opinion_type, 0) + 1
                    total_similarity += matched_op.similarity

            # Stage 3: Generate conclusion
            conclusion = ""
            if request.generate_conclusion and self.conclusion_generator and classified_opinions:
                opinions = [
                    {"text": co.text, "type": co.opinion_type}
                    for co in classified_opinions
                ]
                conclusion = self.conclusion_generator.generate(
                    topic_text=request.topic_text,
                    opinions=opinions
                )

            # Calculate metrics
            matched_count = len(matched_opinions)
            avg_similarity = total_similarity / matched_count if matched_count > 0 else 0.0

            metrics = opinion_service_pb2.AnalysisMetrics(
                total_opinions_analyzed=len(request.all_opinions),
                matched_opinions=matched_count,
                avg_similarity=avg_similarity,
                type_distribution=type_distribution
            )

            return opinion_service_pb2.TopicAnalysisResponse(
                topic_id=request.topic_id,
                matched_opinions=matched_opinions,
                classified_opinions=classified_opinions,
                conclusion=conclusion if conclusion else None,
                metrics=metrics
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Analysis error: {str(e)}")
            return opinion_service_pb2.TopicAnalysisResponse()


def serve(host="localhost", port=50051, max_workers=10):
    """
    Start the gRPC server.

    Args:
        host: Server host
        port: Server port
        max_workers: Thread pool size
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    servicer = OpinionAnalysisServicer()

    opinion_service_pb2_grpc.add_OpinionAnalysisServiceServicer_to_server(servicer, server)

    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)

    print(f"\n{'='*60}")
    print(f"Starting gRPC OpinionAnalysisService on {server_address}")
    print(f"{'='*60}\n")

    server.start()

    try:
        print("Server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(86400)  # One day
    except KeyboardInterrupt:
        print("\nShutting down gRPC server...")
        server.stop(grace=5)
        print("Server stopped.")


if __name__ == "__main__":
    serve()
