"""
FastAPI Gateway for Social Media Analysis
REST API endpoints that interface with the ML pipeline.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import uuid
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings, OPINION_TYPES


# Pydantic models for request/response
class OpinionMatchRequest(BaseModel):
    opinion_text: str = Field(..., min_length=1, description="Opinion text to match")
    top_k: int = Field(default=1, ge=1, le=10, description="Number of top matches")
    threshold: float = Field(default=0.5, ge=0, le=1, description="Similarity threshold")
    
    
class TopicMatch(BaseModel):
    topic_id: str
    topic_text: str
    similarity: float


class OpinionMatchResponse(BaseModel):
    success: bool
    matches: List[TopicMatch]
    error_message: Optional[str] = None


class ClassifyRequest(BaseModel):
    opinion_text: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    success: bool
    predicted_type: Optional[str] = None
    predicted_id: Optional[int] = None
    probabilities: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class Opinion(BaseModel):
    text: str
    type: str = Field(..., description="One of: Claim, Counterclaim, Rebuttal, Evidence")


class ConclusionRequest(BaseModel):
    topic_text: str = Field(..., min_length=1)
    opinions: List[Opinion]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=200, ge=50, le=500)


class ConclusionResponse(BaseModel):
    success: bool
    conclusion: Optional[str] = None
    error_message: Optional[str] = None


class FullAnalysisRequest(BaseModel):
    topic_text: str = Field(..., min_length=1)
    opinion_texts: List[str] = Field(..., min_length=1)
    generate_conclusion: bool = True


class ClassifiedOpinion(BaseModel):
    text: str
    predicted_type: str
    confidence: float


class FullAnalysisResponse(BaseModel):
    success: bool
    topic_text: str
    classified_opinions: List[ClassifiedOpinion] = []
    conclusion: Optional[str] = None
    error_message: Optional[str] = None


class EventRequest(BaseModel):
    """Request to submit analysis as async event."""
    topic_text: str
    opinion_texts: List[str]
    generate_conclusion: bool = True


class EventResponse(BaseModel):
    """Response with event submission details."""
    event_id: str
    correlation_id: str
    status: str = "submitted"


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]


# Create FastAPI app
app = FastAPI(
    title="Social Media Opinion Analyzer",
    description="AI-powered analysis of social media opinions: topic matching, classification, and conclusion generation",
    version="1.0.0"
)


# Global model instances (initialized on startup)
topic_matcher = None
classifier = None
conclusion_generator = None
event_producer = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global topic_matcher, classifier, conclusion_generator, event_producer
    
    # Note: In production, you would load trained models here
    # For now, we'll initialize them lazily when first used
    print("FastAPI server starting...")
    print("Models will be loaded on first request or via /admin/load-models endpoint")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy",
        services={
            "topic_matcher": topic_matcher is not None,
            "classifier": classifier is not None,
            "conclusion_generator": conclusion_generator is not None,
            "event_producer": event_producer is not None
        }
    )


@app.post("/api/v1/match", response_model=OpinionMatchResponse)
async def match_opinion_to_topic(request: OpinionMatchRequest):
    """
    Match an opinion to the most relevant topic(s).
    
    Uses sentence embeddings and cosine similarity.
    """
    global topic_matcher
    
    if topic_matcher is None:
        raise HTTPException(
            status_code=503,
            detail="Topic matcher not initialized. Call /admin/load-models first."
        )
    
    try:
        matches = topic_matcher.match_opinion(
            opinion_text=request.opinion_text,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return OpinionMatchResponse(
            success=True,
            matches=[TopicMatch(**m) for m in matches]
        )
    except Exception as e:
        return OpinionMatchResponse(
            success=False,
            matches=[],
            error_message=str(e)
        )


@app.post("/api/v1/classify", response_model=ClassifyResponse)
async def classify_opinion(request: ClassifyRequest):
    """
    Classify opinion type.
    
    Returns one of: Claim, Counterclaim, Rebuttal, Evidence
    """
    global classifier
    
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not initialized. Call /admin/load-models first."
        )
    
    try:
        result = classifier.predict_single(request.opinion_text)
        
        return ClassifyResponse(
            success=True,
            predicted_type=result["predicted_type"],
            predicted_id=result["predicted_id"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        return ClassifyResponse(
            success=False,
            error_message=str(e)
        )


@app.post("/api/v1/classify/batch", response_model=List[ClassifyResponse])
async def classify_opinions_batch(texts: List[str]):
    """Classify multiple opinions in batch."""
    global classifier
    
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not initialized"
        )
    
    results = []
    for text in texts:
        try:
            result = classifier.predict_single(text)
            results.append(ClassifyResponse(
                success=True,
                predicted_type=result["predicted_type"],
                predicted_id=result["predicted_id"],
                probabilities=result["probabilities"]
            ))
        except Exception as e:
            results.append(ClassifyResponse(
                success=False,
                error_message=str(e)
            ))
    
    return results


@app.post("/api/v1/conclusion", response_model=ConclusionResponse)
async def generate_conclusion(request: ConclusionRequest):
    """
    Generate a conclusion from topic and opinions.
    
    Uses OpenAI API to generate a summary.
    """
    global conclusion_generator
    
    if conclusion_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Conclusion generator not initialized. Set OPENAI_API_KEY and call /admin/load-models."
        )
    
    # Validate opinion types
    valid_types = set(OPINION_TYPES)
    for op in request.opinions:
        if op.type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid opinion type: {op.type}. Must be one of: {OPINION_TYPES}"
            )
    
    try:
        opinions = [{"text": op.text, "type": op.type} for op in request.opinions]
        
        conclusion = conclusion_generator.generate(
            topic_text=request.topic_text,
            opinions=opinions,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ConclusionResponse(
            success=True,
            conclusion=conclusion
        )
    except Exception as e:
        return ConclusionResponse(
            success=False,
            error_message=str(e)
        )


@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
async def full_analysis(request: FullAnalysisRequest):
    """
    Full analysis pipeline:
    1. Classify all opinions
    2. Generate conclusion (optional)
    
    This is a synchronous endpoint. For async processing, use /api/v1/analyze/async
    """
    global classifier, conclusion_generator
    
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not initialized"
        )
    
    try:
        # Classify all opinions
        classified_opinions = []
        for opinion_text in request.opinion_texts:
            result = classifier.predict_single(opinion_text)
            classified_opinions.append(ClassifiedOpinion(
                text=opinion_text,
                predicted_type=result["predicted_type"],
                confidence=max(result["probabilities"].values())
            ))
        
        # Generate conclusion if requested
        conclusion = None
        if request.generate_conclusion and conclusion_generator is not None:
            opinions = [
                {"text": co.text, "type": co.predicted_type}
                for co in classified_opinions
            ]
            conclusion = conclusion_generator.generate(
                topic_text=request.topic_text,
                opinions=opinions
            )
        
        return FullAnalysisResponse(
            success=True,
            topic_text=request.topic_text,
            classified_opinions=classified_opinions,
            conclusion=conclusion
        )
        
    except Exception as e:
        return FullAnalysisResponse(
            success=False,
            topic_text=request.topic_text,
            error_message=str(e)
        )


@app.post("/api/v1/analyze/async", response_model=EventResponse)
async def async_analysis(request: EventRequest, background_tasks: BackgroundTasks):
    """
    Submit analysis request to Redis queue for async processing.
    
    Returns immediately with event_id and correlation_id.
    Use /api/v1/results/{correlation_id} to retrieve results.
    """
    global event_producer
    
    if event_producer is None:
        # Try to initialize
        try:
            from events import EventProducer
            event_producer = EventProducer()
            event_producer.connect()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Event producer not available: {e}"
            )
    
    correlation_id = str(uuid.uuid4())
    
    event_id = event_producer.publish_full_analysis_request(
        topic_text=request.topic_text,
        opinion_texts=request.opinion_texts,
        generate_conclusion=request.generate_conclusion,
        correlation_id=correlation_id
    )
    
    return EventResponse(
        event_id=event_id,
        correlation_id=correlation_id,
        status="submitted"
    )


@app.get("/api/v1/results/{correlation_id}")
async def get_results(correlation_id: str, timeout: int = 10):
    """
    Retrieve results for an async analysis request.
    
    Args:
        correlation_id: The correlation ID returned from /api/v1/analyze/async
        timeout: Seconds to wait for results
    """
    global event_producer
    
    if event_producer is None:
        raise HTTPException(status_code=503, detail="Event producer not available")
    
    result = event_producer.get_result(correlation_id, timeout=timeout)
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Results not found or not ready yet"
        )
    
    return result


# Admin endpoints
@app.post("/admin/load-models")
async def load_models(
    load_topic_matcher: bool = False,
    load_classifier: bool = True,
    load_conclusion_generator: bool = True,
    topic_matcher_path: Optional[str] = None,
    classifier_path: Optional[str] = None
):
    """
    Load ML models into memory.
    
    This endpoint should be called once after server startup.
    """
    global topic_matcher, classifier, conclusion_generator
    
    results = {}
    
    if load_topic_matcher:
        try:
            from models import TopicMatcher
            topic_matcher = TopicMatcher()
            topic_matcher.load_model()
            
            if topic_matcher_path:
                topic_matcher.load(Path(topic_matcher_path))
            
            results["topic_matcher"] = "loaded"
        except Exception as e:
            results["topic_matcher"] = f"error: {e}"
    
    if load_classifier:
        try:
            from models import OpinionClassifier
            classifier = OpinionClassifier()
            
            if classifier_path:
                classifier.load(Path(classifier_path))
            else:
                classifier.load_model()
            
            results["classifier"] = "loaded"
        except Exception as e:
            results["classifier"] = f"error: {e}"
    
    if load_conclusion_generator:
        try:
            from models import ConclusionGenerator
            conclusion_generator = ConclusionGenerator()
            conclusion_generator.initialize()
            results["conclusion_generator"] = "loaded"
        except Exception as e:
            results["conclusion_generator"] = f"error: {e}"
    
    return {"status": "complete", "results": results}


@app.get("/")
async def root():
    """API root with documentation links."""
    return {
        "message": "Social Media Opinion Analyzer API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )