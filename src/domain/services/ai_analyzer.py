from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..entities.commit import Commit
from ..entities.analysis import AnalysisResult


class IAIAnalyzer(ABC):
    @abstractmethod
    async def analyzer_commit(self, commit: Commit) -> AnalysisResult:
        """Analyze a commit and return the analysis result"""
        pass

    @abstractmethod
    async def generate_summary(self, commits: List[Commit]) -> str:
        """Generate a summary of the analysis results for a list of commits"""
        pass

    @abstractmethod
    async def answer_question(
            self,
            question: str,
            context_commits: List[Commit]
    ) -> str:
        """Answer question about commits"""
        pass

    @abstractmethod
    async def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        pass
