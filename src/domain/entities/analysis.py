from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from .base import Entity, ValueObject


@dataclass
class AnalysisResult(ValueObject):
    """Value object for AI analysis results"""
    summary: str
    tags: List[str]
    sentiment_score: float
    confidence_score: float
    extracted_entities: List[str]

    def _validate(self):
        if not (-1 <= self.sentiment_score <= 1):
            raise ValueError("Sentiment score must be between -1 and 1")
        if not (0 <= self.confidence_score <= 1):
            raise ValueError("Confidence score must be between 0 and 1")


class Analysis(Entity):
    def __init__(
        self,
        commit_id: str,
        model_name: str,
        model_version: str,
        prompt_version: str,
        result: AnalysisResult,
        processing_time_ms: int,
        tokens_used: int = 0
    ):
        super().__init__()
        self.commit_id: str = commit_id
        self.model_name: str = model_name
        self.model_version: str = model_version
        self.prompt_version: str = prompt_version
        self.result: AnalysisResult = result
        self.processing_time_ms: int = processing_time_ms
        self.tokens_used: int = tokens_used
        self.status = "completed"  # Default status

    def is_high_confidence(self) -> bool:
        """Check if analysis has high confidence"""
        return self.result.confidence_score >= 0.8
