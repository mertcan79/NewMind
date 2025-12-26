"""
CONFIGURATION SETTINGS FOR SOCIAL MEDIA ANALYSIS SYSTEM
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

# LOAD ENVIRONMENT VARIABLES FROM .ENV FILE
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """
    APPLICATION SETTINGS LOADED FROM ENVIRONMENT VARIABLES
    """

    # PATHS
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "trained_models"

    # REDIS
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_QUEUE_NAME: str = "social_media_events"
    REDIS_RESULT_QUEUE: str = "social_media_results"

    # GRPC
    GRPC_HOST: str = "localhost"
    GRPC_PORT: int = 50051

    # OPENAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # MODEL SETTINGS
    TOPIC_MATCHER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLASSIFIER_MODEL: str = "distilbert-base-uncased"
    SIMILARITY_THRESHOLD: float = 0.5

    # TRAINING
    TRAIN_BATCH_SIZE: int = 16
    EVAL_BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5
    MAX_SEQ_LENGTH: int = 256

    # DATA SPLIT
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    RANDOM_SEED: int = 42

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# GLOBAL SETTINGS INSTANCE
settings = Settings()


# OPINION TYPE LABELS
OPINION_TYPES = ["Claim", "Counterclaim", "Rebuttal", "Evidence"]
OPINION_TYPE_TO_ID = {label: idx for idx, label in enumerate(OPINION_TYPES)}
ID_TO_OPINION_TYPE = {idx: label for idx, label in enumerate(OPINION_TYPES)}
