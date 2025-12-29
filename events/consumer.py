"""
Redis Event Consumer/Worker
Processes opinion analysis events from Redis.
"""
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import redis

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from models import ConclusionGenerator, OpinionClassifier, TopicMatcher


class EventConsumer:
    """
    Consumes and processes events from Redis queue.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        queue_name: str = None,
        result_queue: str = None
    ):
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.queue_name = queue_name or settings.REDIS_QUEUE_NAME
        self.result_queue = result_queue or settings.REDIS_RESULT_QUEUE

        self.client = None
        self.running = False

        # Models
        self.topic_matcher: Optional[TopicMatcher] = None
        self.classifier: Optional[OpinionClassifier] = None
        self.conclusion_generator: Optional[ConclusionGenerator] = None

    def connect(self):
        """Connect to Redis."""
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True
        )
        self.client.ping()
        print(f"Connected to Redis at {self.host}:{self.port}")

    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()

    def initialize_models(
        self,
        topic_matcher: TopicMatcher = None,
        classifier: OpinionClassifier = None,
        conclusion_generator: ConclusionGenerator = None
    ):
        """Set the ML models to use for processing."""
        self.topic_matcher = topic_matcher
        self.classifier = classifier
        self.conclusion_generator = conclusion_generator

        print("Models initialized:")
        print(f"  Topic Matcher: {'Yes' if topic_matcher else 'No'}")
        print(f"  Classifier: {'Yes' if classifier else 'No'}")
        print(f"  Conclusion Generator: {'Yes' if conclusion_generator else 'No'}")

    def _publish_result(self, correlation_id: str, result: Dict):
        """Publish result to correlation-specific queue."""
        result_key = f"{self.result_queue}:{correlation_id}"
        self.client.lpush(result_key, json.dumps(result))
        # Set expiry for 5 minutes
        self.client.expire(result_key, 300)

    def _process_match_request(self, payload: Dict) -> Dict:
        """Process topic matching request."""

        if not self.topic_matcher:
            return {"success": False, "error": "Topic matcher not initialized"}

        # Note: Topic matching requires pre-loaded topics
        # For full matching implementation, use the gRPC service
        return {
            "success": False,
            "error": "Topic matching requires pre-loaded topics. Use gRPC service for matching."
        }

    def _process_classify_request(self, payload: Dict) -> Dict:
        """Process classification request."""

        if not self.classifier:
            return {"success": False, "error": "Classifier not initialized"}

        result = self.classifier.predict(payload["opinion_text"])

        return {
            "success": True,
            "predicted_type": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

    def _process_conclusion_request(self, payload: Dict) -> Dict:
        """Process conclusion generation request."""

        if not self.conclusion_generator:
            return {"success": False, "error": "Conclusion generator not initialized"}

        conclusion = self.conclusion_generator.generate(
            topic_text=payload["topic_text"],
            opinions=payload["opinions"],
            temperature=payload.get("temperature", 0.7),
            max_tokens=payload.get("max_tokens", 200)
        )

        return {
            "success": True,
            "conclusion": conclusion
        }

    def _process_full_analysis(self, payload: Dict) -> Dict:
        """Process full analysis request."""

        result = {
            "success": True,
            "topic_text": payload["topic_text"],
            "classified_opinions": [],
            "conclusion": ""
        }

        # Classify opinions
        if self.classifier:
            for opinion_text in payload["opinion_texts"]:
                classification = self.classifier.predict_single(opinion_text)
                result["classified_opinions"].append({
                    "text": opinion_text,
                    "predicted_type": classification["predicted_type"],
                    "confidence": max(classification["probabilities"].values())
                })

        # Generate conclusion
        if payload.get("generate_conclusion") and self.conclusion_generator:
            opinions = [
                {"text": co["text"], "type": co["predicted_type"]}
                for co in result["classified_opinions"]
            ]

            result["conclusion"] = self.conclusion_generator.generate(
                topic_text=payload["topic_text"],
                opinions=opinions
            )

        return result

    def process_event(self, event: Dict) -> Dict:
        """Process a single event."""

        event_type = event.get("event_type")
        payload = event.get("payload", {})

        processors = {
            "MATCH_OPINION_TO_TOPIC": self._process_match_request,
            "CLASSIFY_OPINION": self._process_classify_request,
            "GENERATE_CONCLUSION": self._process_conclusion_request,
            "FULL_ANALYSIS": self._process_full_analysis
        }

        processor = processors.get(event_type)

        if not processor:
            return {"success": False, "error": f"Unknown event type: {event_type}"}

        try:
            result = processor(payload)
            result["event_id"] = event.get("event_id")
            result["processed_at"] = datetime.utcnow().isoformat()
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "event_id": event.get("event_id")
            }

    def run(self, blocking_timeout: int = 5):
        """
        Start consuming events from the queue.
        
        Args:
            blocking_timeout: Seconds to wait for new events
        """
        self.running = True

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down consumer...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"Starting event consumer on queue: {self.queue_name}")

        while self.running:
            try:
                # Blocking pop with timeout
                item = self.client.brpop(self.queue_name, timeout=blocking_timeout)

                if item:
                    queue_name, event_data = item
                    event = json.loads(event_data)

                    print(f"Processing event: {event.get('event_type')} [{event.get('event_id')}]")

                    # Process the event
                    result = self.process_event(event)

                    # Publish result if correlation_id exists
                    correlation_id = event.get("payload", {}).get("correlation_id")
                    if correlation_id:
                        self._publish_result(correlation_id, result)

                    print(f"  Result: {'Success' if result.get('success') else 'Failed'}")

            except redis.ConnectionError as e:
                print(f"Redis connection error: {e}")
                if self.running:
                    import time
                    time.sleep(5)  # Wait before reconnecting
                    self.connect()
            except Exception as e:
                print(f"Error processing event: {e}")

        print("Consumer stopped")


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    load_dotenv()

    # Create consumer
    consumer = EventConsumer()

    try:
        consumer.connect()

        # Initialize models
        print("\nInitializing models...")

        classifier = OpinionClassifier()
        print("✓ Classifier loaded")

        topic_matcher = TopicMatcher()
        topic_matcher.load_model()
        print("✓ Topic Matcher loaded")

        conclusion_generator = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            conclusion_generator = ConclusionGenerator(api_key=api_key)
            conclusion_generator.initialize()
            print("✓ Conclusion Generator loaded")
        else:
            print("⚠  OPENAI_API_KEY not set - conclusion generation disabled")

        consumer.initialize_models(
            topic_matcher=topic_matcher,
            classifier=classifier,
            conclusion_generator=conclusion_generator
        )

        print("\nStarting consumer (Ctrl+C to stop)...")
        consumer.run()

    except redis.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        print("Make sure Redis is running on localhost:6379")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        consumer.close()
