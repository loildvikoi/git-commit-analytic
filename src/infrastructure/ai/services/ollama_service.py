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
import asyncio

logger = logging.getLogger(__name__)


class OllamaService(IAIAnalyzer):
    """Ollama AI service implementation with retry and error handling"""

    def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 120,
            max_retries: int = 2,
            retry_delay: int = 3
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.prompts = CommitPrompts()

        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')

        # HTTP client configuration
        self.client_config = {
            'timeout': httpx.Timeout(
                connect=10.0,   # Connect timeout
                read=120.0,     # Read timeout cao
                write=30.0,     # Write timeout
                pool=30.0       # Pool timeout
            ),
            'limits': httpx.Limits(max_connections=2, max_keepalive_connections=1)
        }

    async def analyzer_commit(self, commit: Commit) -> AnalysisResult:
        """Analyze a commit using Ollama with retry logic"""
        prompt = self.prompts.get_commit_analysis_prompt(commit)

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)

                # Parse structured response
                analysis_data = self._parse_analysis_response(response)

                result = AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=float(analysis_data.get('sentiment', 0.0)),
                    confidence_score=float(analysis_data.get('confidence', 0.5)),
                    extracted_entities=analysis_data.get('entities', [])
                )

                logger.info(f"Successfully analyzed commit {commit.commit_hash.value} in {processing_time}ms")
                return result

            except Exception as e:
                logger.warning(f"Analysis attempt {attempt + 1} failed for {commit.commit_hash.value}: {str(e)}")

                if attempt == self.max_retries - 1:
                    # Last attempt failed, return error result
                    logger.error(f"All analysis attempts failed for {commit.commit_hash.value}")
                    return AnalysisResult(
                        summary=f"Analysis failed after {self.max_retries} attempts: {str(e)}",
                        tags=['error', 'analysis_failed'],
                        sentiment_score=0.0,
                        confidence_score=0.0,
                        extracted_entities=[]
                    )

                # Wait before retry
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def generate_summary(self, commits: List[Commit]) -> str:
        """Generate summary for multiple commits"""
        if not commits:
            return "No commits to summarize."

        if len(commits) == 1:
            analysis = await self.analyze_commit(commits[0])
            return analysis.summary

        prompt = self.prompts.get_batch_summary_prompt(commits)

        try:
            response = await self._generate_response(prompt, temperature=0.5)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating batch summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    async def answer_question(self, question: str, context_commits: List[Commit]) -> str:
        """Answer question about commits"""
        if not context_commits:
            return "No commits found in the specified context."

        prompt = self.prompts.get_question_answer_prompt(question, context_commits)
        logger.info(f"Answering question: {question}")
        logger.info(f"Promt for question: {prompt}")

        try:
            response = await self._generate_response(prompt, temperature=0.7)
            return response.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    async def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        prompt = self.prompts.get_entity_extraction_prompt(text)

        try:
            response = await self._generate_response(prompt, temperature=0.3)
            entities_data = self._parse_json_response(response)
            return entities_data.get('entities', [])
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            async with httpx.AsyncClient(**self.client_config) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            'name': self.model,
            'provider': 'ollama',
            'base_url': self.base_url,
            'version': 'latest'
        }

    async def _generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate response from Ollama with proper error handling"""
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens,  # Giảm max tokens
                'top_k': 20,        # Giảm từ 40
                'top_p': 0.8,       # Giảm từ 0.9
                'repeat_penalty': 1.05,  # Giảm từ 1.1
                'num_thread': 4,    # Giới hạn thread
                'num_ctx': 2048,    # Giảm context window
                # 'batch_size': 1,    # Process từng batch nhỏ
                # 'low_vram': True    # Tối ưu memory
            }
        }

        async with httpx.AsyncClient(**self.client_config) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()

                result = response.json()
                if 'response' not in result:
                    raise ValueError("Invalid response format from Ollama")

                return result['response']

            except httpx.TimeoutException:
                raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
            except httpx.HTTPStatusError as e:
                raise ConnectionError(f"Ollama HTTP error: {e.response.status_code}")
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {str(e)}")

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from analysis with fallback"""
        try:
            # Try to parse as JSON first
            return self._parse_json_response(response)
        except:
            # Fallback to text parsing
            return self._parse_text_response(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response"""
        # Clean up response text
        response = response.strip()

        # Find JSON block
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON found in response")

        json_str = response[start_idx:end_idx + 1]
        return json.loads(json_str)

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """Fallback text parsing when JSON parsing fails"""
        logger.warning("Falling back to text parsing for Ollama response")

        # Default values
        result = {
            'summary': response[:200] + '...' if len(response) > 200 else response,
            'tags': [],
            'sentiment': 0.0,
            'confidence': 0.3,
            'entities': []
        }

        # Simple text extraction
        lines = response.lower().split('\n')

        # Extract sentiment indicators
        if any(word in response.lower() for word in ['fix', 'bug', 'error', 'issue']):
            result['sentiment'] = -0.2
            result['tags'].append('bugfix')
        elif any(word in response.lower() for word in ['feature', 'add', 'new', 'implement']):
            result['sentiment'] = 0.3
            result['tags'].append('feature')
        elif any(word in response.lower() for word in ['refactor', 'improve', 'optimize']):
            result['sentiment'] = 0.1
            result['tags'].append('refactor')

        return result
