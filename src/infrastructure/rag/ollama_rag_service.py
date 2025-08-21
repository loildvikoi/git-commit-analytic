from typing import List, Dict, Any, Optional
import logging
import re
from ...domain.services.rag_service import IRAGService, RAGContext
from ...domain.services.search_service import ISearchService, SearchQuery
from ...domain.services.ai_analyzer import IAIAnalyzer
from ...domain.entities.document import Document

logger = logging.getLogger(__name__)


class OllamaRAGService(IRAGService):
    """RAG service using Ollama for generation"""

    def __init__(
            self,
            search_service: ISearchService,
            ai_analyzer: IAIAnalyzer,
            max_context_tokens: int = 2000,
            temperature: float = 0.3
    ):
        self.search_service = search_service
        self.ai_analyzer = ai_analyzer
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature

    async def answer_question(
            self,
            question: str,
            context_documents: Optional[List[Document]] = None,
            search_first: bool = True,
            max_documents: int = 5
    ) -> Dict[str, Any]:
        """Answer a question using RAG approach"""

        try:
            # Get context documents if not provided
            if context_documents is None and search_first:
                context = await self.generate_context(question, max_documents)
                context_documents = context.documents
            elif context_documents:
                context = RAGContext(
                    question=question,
                    documents=context_documents,
                    max_context_length=self.max_context_tokens
                )
            else:
                # No context available
                return {
                    "answer": "I don't have enough context to answer this question.",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "no_context"
                }

            # Build context text
            context_text = context.build_context_text()

            if not context_text:
                return {
                    "answer": "No relevant information found to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "method": "no_relevant_context"
                }

            # Create RAG prompt
            prompt = self._create_rag_prompt(question, context_text)

            # Get answer from AI
            answer = await self._generate_answer(prompt)

            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(context_documents, answer)

            return {
                "answer": answer,
                "sources": [doc.id for doc in context_documents],
                "confidence": confidence,
                "method": "rag_with_context",
                "context_length": len(context_text),
                "documents_used": len(context_documents)
            }

        except Exception as e:
            logger.error(f"Error in RAG answer_question: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "method": "error"
            }

    async def generate_context(
            self,
            question: str,
            max_documents: int = 5
    ) -> RAGContext:
        """Generate context for a question by finding relevant documents"""

        # Search for relevant documents
        search_query = SearchQuery(
            text=question,
            use_hybrid=True,
            use_reranking=True,
            max_results=max_documents * 2  # Get more for selection
        )

        search_results = await self.search_service.search(search_query)

        # Select top documents based on relevance
        selected_documents = []
        for result in search_results[:max_documents]:
            if result.score.combined_score > 0.3:  # Minimum relevance threshold
                selected_documents.append(result.document)

        # Create context
        context = RAGContext(
            question=question,
            documents=selected_documents,
            max_context_length=self.max_context_tokens
        )

        return context

    async def augment_query(self, query: str) -> List[str]:
        """Augment query with synonyms, related terms, etc."""

        augmented = [query]

        # Add variations based on common patterns
        # Extract potential keywords
        keywords = await self.extract_keywords(query)

        # Add synonym variations
        synonyms = self._get_synonyms(query)
        augmented.extend(synonyms)

        # Add question variations
        if "what" in query.lower():
            augmented.append(query.replace("what", "which"))
        if "how" in query.lower():
            augmented.append(query.replace("how", "what way"))

        # Add keyword combinations
        if len(keywords) > 1:
            augmented.append(" ".join(keywords))

        # Remove duplicates and return
        return list(set(augmented))

    async def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""

        # Remove common words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'in', 'out', 'up', 'down', 'off', 'over', 'under', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'by', 'for', 'from', 'of', 'to', 'with'
        }

        # Tokenize and filter
        words = text.lower().split()
        keywords = []

        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s]', '', word)

            # Keep if not a stop word and has meaningful length
            if word and word not in stop_words and len(word) > 2:
                keywords.append(word)

        # Also extract potential technical terms (CamelCase, snake_case, etc.)
        technical_terms = re.findall(r'[A-Z][a-z]+|[A-Z]+|[a-z]+_[a-z]+', text)
        keywords.extend([term.lower() for term in technical_terms])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords[:10]  # Return top 10 keywords

    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a RAG prompt for the AI"""

        prompt = f"""You are an AI assistant helping developers understand their codebase and development history.

Use the following context to answer the question. If you cannot answer the question based on the context, say so.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and reference details from the context
3. If the context doesn't contain the answer, say "I don't have enough information in the provided context"
4. Keep the answer concise but complete
5. Use technical terms appropriately

ANSWER:"""

        return prompt

    async def _generate_answer(self, prompt: str) -> str:
        """Generate answer using AI service"""

        # Since IAIAnalyzer doesn't have a direct generation method,
        # we'll use answer_question with empty context
        # In a real implementation, you might want to add a generate method to IAIAnalyzer

        # For now, we'll use a workaround
        from ...domain.entities.commit import Commit, CommitHash
        from datetime import datetime

        # Create a dummy commit with our prompt as the message
        dummy_commit = Commit(
            commit_hash=CommitHash("dummy123456"),
            author_email="system@rag",
            author_name="RAG System",
            message=prompt,
            timestamp=datetime.now(),
            branch="main",
            project="rag",
            files_changed=[]
        )

        # Use the AI analyzer to generate a response
        result = await self.ai_analyzer.answer_question(
            question="Generate response",
            context_commits=[dummy_commit]
        )

        return result

    def _calculate_confidence(
            self,
            documents: List[Document],
            answer: str
    ) -> float:
        """Calculate confidence score for the answer"""

        if not documents:
            return 0.0

        confidence = 0.5  # Base confidence

        # Increase confidence based on number of relevant documents
        if len(documents) >= 5:
            confidence += 0.2
        elif len(documents) >= 3:
            confidence += 0.1

        # Check if answer seems complete
        if len(answer) > 100:
            confidence += 0.1

        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i don't", "not sure", "unclear", "might be", "possibly",
            "don't have enough", "cannot determine"
        ]

        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                confidence -= 0.3
                break

        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))

    def _get_synonyms(self, text: str) -> List[str]:
        """Get synonyms for query expansion"""

        # Simple synonym mapping for common development terms
        synonym_map = {
            "bug": ["issue", "error", "problem", "defect"],
            "fix": ["repair", "resolve", "patch", "correct"],
            "feature": ["functionality", "capability", "enhancement"],
            "commit": ["change", "revision", "update"],
            "code": ["source", "implementation", "logic"],
            "test": ["testing", "verification", "validation"],
            "review": ["examine", "check", "inspect"],
            "merge": ["combine", "integrate", "join"],
            "branch": ["fork", "version", "variant"],
            "deploy": ["release", "publish", "launch"]
        }

        variations = []
        text_lower = text.lower()

        for term, synonyms in synonym_map.items():
            if term in text_lower:
                for synonym in synonyms:
                    variations.append(text_lower.replace(term, synonym))

        return variations[:3]  # Return top 3 variations