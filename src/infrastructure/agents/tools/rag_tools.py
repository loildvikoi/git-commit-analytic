# src/infrastructure/agents/tools/rag_tools.py
"""
Tools that give agents access to Phase 2 RAG capabilities
"""
from typing import Dict, Any, List
from langchain.tools import Tool
import logging

logger = logging.getLogger(__name__)


class RAGTools:
    """Tools that integrate Phase 2 RAG with Phase 3 Agents"""

    def __init__(self, search_service, rag_service, document_repository):
        self.search_service = search_service
        self.rag_service = rag_service
        self.document_repository = document_repository

    def get_tools(self) -> List[Tool]:
        """Get all RAG tools for agents"""
        return [
            Tool(
                name="semantic_search",
                func=self.semantic_search_tool,
                description="Search documents using semantic similarity. Input: search query"
            ),
            Tool(
                name="rag_query",
                func=self.rag_query_tool,
                description="Answer questions using RAG. Input: question"
            ),
            Tool(
                name="find_similar_commits",
                func=self.find_similar_commits_tool,
                description="Find commits similar to given text. Input: commit message or description"
            ),
            Tool(
                name="get_project_context",
                func=self.get_project_context_tool,
                description="Get context about a project. Input: project name"
            )
        ]

    async def semantic_search_tool(self, query: str) -> str:
        """Tool for semantic search using Phase 2 infrastructure"""
        try:
            from ...domain.services.search_service import SearchQuery

            search_query = SearchQuery(
                text=query,
                use_hybrid=True,
                use_reranking=True,
                max_results=5
            )

            results = await self.search_service.search(search_query)

            # Format results for agent
            formatted = "Search Results:\n\n"
            for i, result in enumerate(results, 1):
                formatted += f"{i}. Score: {result.score.combined_score:.2f}\n"
                formatted += f"   Content: {result.document.content[:200]}...\n"
                formatted += f"   Project: {result.document.project}\n\n"

            return formatted

        except Exception as e:
            logger.error(f"Semantic search tool error: {str(e)}")
            return f"Search failed: {str(e)}"

    async def rag_query_tool(self, question: str) -> str:
        """Tool for RAG queries using Phase 2 infrastructure"""
        try:
            result = await self.rag_service.answer_question(
                question=question,
                search_first=True,
                max_documents=5
            )

            answer = result.get("answer", "No answer found")
            confidence = result.get("confidence", 0)
            sources_count = len(result.get("sources", []))

            return f"Answer (confidence: {confidence:.2f}, sources: {sources_count}):\n{answer}"

        except Exception as e:
            logger.error(f"RAG query tool error: {str(e)}")
            return f"Query failed: {str(e)}"

    async def find_similar_commits_tool(self, text: str) -> str:
        """Find similar commits using embeddings"""
        try:
            # Use Phase 2 semantic search
            from ...domain.services.search_service import SearchQuery
            from ...domain.entities.document import DocumentType

            search_query = SearchQuery(
                text=text,
                document_types=[DocumentType.COMMIT],
                use_hybrid=True,
                max_results=5
            )

            results = await self.search_service.search(search_query)

            formatted = "Similar Commits:\n\n"
            for result in results:
                formatted += f"- {result.document.title}\n"
                formatted += f"  Project: {result.document.project}\n"
                formatted += f"  Similarity: {result.score.combined_score:.2f}\n\n"

            return formatted

        except Exception as e:
            return f"Search failed: {str(e)}"

    async def get_project_context_tool(self, project_name: str) -> str:
        """Get context about a project from documents"""
        try:
            # Search for project documents
            from ...domain.entities.document import DocumentType

            documents = await self.document_repository.search(
                query="",
                project=project_name,
                limit=10
            )

            if not documents:
                return f"No information found for project: {project_name}"

            # Summarize project activity
            context = f"Project: {project_name}\n\n"
            context += f"Total documents: {len(documents)}\n"

            # Get recent commits
            recent_commits = [d for d in documents if d.document_type == DocumentType.COMMIT]
            if recent_commits:
                context += f"Recent commits: {len(recent_commits)}\n"
                context += "Latest changes:\n"
                for commit in recent_commits[:3]:
                    context += f"- {commit.title}\n"

            return context

        except Exception as e:
            return f"Failed to get project context: {str(e)}"
