"""
REDIS EVENT PRODUCER
Publishes opinion analysis events to Redis queue.
"""
import redis
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class EventProducer:
    """
    PUBLISHES EVENTS TO REDIS QUEUE FOR ASYNC PROCESSING
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

    def connect(self):
        """ESTABLISH CONNECTION TO REDIS"""
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True
        )
        self.client.ping()
        print(f"Connected to Redis at {self.host}:{self.port}")

    def close(self):
        """CLOSE REDIS CONNECTION"""
        if self.client:
            self.client.close()

    def _create_event(self, event_type: str, payload: Dict) -> Dict:
        """CREATE A STANDARD EVENT STRUCTURE"""
        return {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload
        }

    def _publish(self, event: Dict) -> str:
        """PUBLISH EVENT TO QUEUE"""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        self.client.lpush(self.queue_name, json.dumps(event))
        return event["event_id"]

    def publish_match_request(
        self,
        opinion_text: str,
        top_k: int = 1,
        threshold: float = 0.5,
        correlation_id: str = None
    ) -> str:
        """
        PUBLISH A TOPIC MATCHING REQUEST

        Args:
            opinion_text: The opinion to match to topics
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            correlation_id: Optional ID for tracking results

        Returns:
            Event ID
        """
        payload = {
            "opinion_text": opinion_text,
            "top_k": top_k,
            "threshold": threshold
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id

        event = self._create_event("MATCH_OPINION_TO_TOPIC", payload)
        return self._publish(event)

    def publish_classify_request(
        self,
        opinion_text: str,
        correlation_id: str = None
    ) -> str:
        """
        PUBLISH A CLASSIFICATION REQUEST

        Args:
            opinion_text: The opinion to classify
            correlation_id: Optional ID for tracking results

        Returns:
            Event ID
        """
        payload = {"opinion_text": opinion_text}
        if correlation_id:
            payload["correlation_id"] = correlation_id

        event = self._create_event("CLASSIFY_OPINION", payload)
        return self._publish(event)

    def publish_conclusion_request(
        self,
        topic_text: str,
        opinions: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 200,
        correlation_id: str = None
    ) -> str:
        """
        PUBLISH A CONCLUSION GENERATION REQUEST

        Args:
            topic_text: The main topic text
            opinions: List of dicts with 'text' and 'type' keys
            temperature: OpenAI temperature setting
            max_tokens: Maximum tokens in response
            correlation_id: Optional ID for tracking results

        Returns:
            Event ID
        """
        payload = {
            "topic_text": topic_text,
            "opinions": opinions,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id

        event = self._create_event("GENERATE_CONCLUSION", payload)
        return self._publish(event)

    def publish_full_analysis_request(
        self,
        topic_text: str,
        opinion_texts: List[str],
        generate_conclusion: bool = True,
        correlation_id: str = None
    ) -> str:
        """
        PUBLISH A FULL ANALYSIS REQUEST

        Args:
            topic_text: The main topic text
            opinion_texts: List of opinion texts to classify
            generate_conclusion: Whether to generate conclusion after classification
            correlation_id: Optional ID for tracking results

        Returns:
            Event ID
        """
        payload = {
            "topic_text": topic_text,
            "opinion_texts": opinion_texts,
            "generate_conclusion": generate_conclusion
        }
        if correlation_id:
            payload["correlation_id"] = correlation_id

        event = self._create_event("FULL_ANALYSIS", payload)
        return self._publish(event)

    def get_result(self, correlation_id: str, timeout: int = 10) -> Optional[Dict]:
        """
        RETRIEVE RESULT BY CORRELATION ID

        Args:
            correlation_id: The correlation ID to look up
            timeout: Seconds to wait for result

        Returns:
            Result dict or None if not found/timeout
        """
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        result_key = f"{self.result_queue}:{correlation_id}"

        # BLOCKING POP WITH TIMEOUT
        result = self.client.brpop(result_key, timeout=timeout)

        if result:
            _, data = result
            return json.loads(data)
        return None


if __name__ == "__main__":
    # TEST THE PRODUCER
    producer = EventProducer()

    try:
        producer.connect()

        # PUBLISH A TEST EVENT
        event_id = producer.publish_classify_request(
            opinion_text="I think this is true because of the evidence",
            correlation_id="test-123"
        )
        print(f"Published event: {event_id}")

        # PUBLISH FULL ANALYSIS
        event_id = producer.publish_full_analysis_request(
            topic_text="Climate change is real",
            opinion_texts=[
                "Scientific data supports this",
                "Some disagree with the methodology"
            ],
            correlation_id="test-456"
        )
        print(f"Published full analysis event: {event_id}")

    except redis.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        print("Make sure Redis is running on localhost:6379")
    finally:
        producer.close()
