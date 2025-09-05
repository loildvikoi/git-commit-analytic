
# src/application/use_cases/agent_enhanced_chat.py
"""
Enhanced chat that uses agents with RAG
"""
from typing import Optional, Any, Dict

from src.domain.entities.agent import AgentRole
from src.domain.services.agent_service import IAgentService
from src.domain.services.rag_service import IRAGService
from src.domain.services.search_service import ISearchService


class AgentEnhancedChatUseCase:
    """Chat enhanced with agent capabilities"""

    def __init__(
            self,
            agent_service: IAgentService,
            rag_service: IRAGService,
            search_service: ISearchService
    ):
        self.agent_service = agent_service
        self.rag_service = rag_service
        self.search_service = search_service

    async def execute(
            self,
            question: str,
            use_agent: bool = True,
            context_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer question using agent + RAG"""

        if use_agent:
            # Create assistant agent with RAG tools
            agent = await self.agent_service.create_agent(
                name="Assistant",
                role=AgentRole.GENERAL_ASSISTANT,
                model="llama3.2:1b"
            )

            # Give agent access to RAG tools
            from ...infrastructure.agents.tools.rag_tools import RAGTools
            rag_tools = RAGTools(
                self.search_service,
                self.rag_service,
                None  # document repo
            )

            # Agent will use tools to answer
            result = await self.agent_service.execute_task(
                agent=agent,
                task=question,
                context={
                    "project": context_project,
                    "available_tools": ["semantic_search", "rag_query"]
                }
            )

            return {
                "answer": result.get("output", ""),
                "method": "agent_with_rag",
                "tools_used": agent.tools_used,
                "confidence": 0.9
            }
        else:
            # Fallback to Phase 2 RAG
            return await self.rag_service.answer_question(
                question=question,
                search_first=True
            )