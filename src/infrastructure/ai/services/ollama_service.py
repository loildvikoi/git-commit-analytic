import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from ....domain.services.ai_analyzer import IAIAnalyzer
from ....domain.entities.commit import Commit
from ....domain.entities.analysis import AnalysisResult
from .prompts.commit_prompts import CommitPrompts
import time
import os

logger = logging.getLogger(__name__)


class OllamaService(IAIAnalyzer):
    """Ollama AI service implementation"""

    def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout
        self.prompts = CommitPrompts()

        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')

    async def analyze_commit(self, commit: Commit) -> AnalysisResult:
        """Analyze a commit using Ollama"""
        prompt = self.prompts.get_commit_analysis_prompt(commit)

        try:
            start_time = time.time()
            response = await self._generate_response(prompt, temperature=0.3)
            processing_time = int((time.time() - start_time) * 1000)

            # Parse structured response
            analysis_data = self._parse_analysis_response(response)

            return AnalysisResult(
                summary=analysis_data.get('summary', 'No summary available'),
                tags=analysis_data.get('tags', []),
                sentiment_score=analysis_data.get('sentiment', 0.0),
                confidence_score=analysis_data.get('confidence', 0.5),
                extracted_entities=analysis_data.get('entities', [])
            )

        except Exception as e:
            logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
            # Return default analysis on error
            return AnalysisResult(
                summary=f"Analysis failed: {str(e)}",
                tags=['error'],
                sentiment_score=0.0,
                confidence_score=0.0,
                extracted_entities=[]
            )

    async def generate_summary(self, commits: List[Commit]) -> str:
        """Generate summary for multiple commits"""
        if not commits:
            return "No commits to summarize."
