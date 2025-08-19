from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ...domain.entities.analysis import AnalysisResult


class IAIService(ABC):
    """Interface for AI service"""

    @abstractmethod
    async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
    ) -> AnalysisResult:
        """Analyze text and return structured results"""
        pass

    @abstractmethod
    async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
    ) -> str:
        """Generate text response"""
        pass

    @abstractmethod
    async def create_embedding(self, text: str) -> List[float]:
        """Create text embedding"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if AI service is healthy"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        pass
