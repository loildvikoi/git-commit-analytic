# src/domain/services/agent_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..entities.agent import Agent, AgentRole, AgentDecision


class IAgentService(ABC):
    """Service interface for agent operations"""

    @abstractmethod
    async def create_agent(
            self,
            name: str,
            role: AgentRole,
            model: str = "llama3.2:1b"
    ) -> Agent:
        """Create a new agent"""
        pass

    @abstractmethod
    async def execute_task(
            self,
            agent: Agent,
            task: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task with an agent"""
        pass

    @abstractmethod
    async def get_agent_decision(
            self,
            agent: Agent,
            situation: str,
            options: List[str]
    ) -> AgentDecision:
        """Get agent's decision for a situation"""
        pass

    @abstractmethod
    async def collaborate(
            self,
            agents: List[Agent],
            task: str
    ) -> Dict[str, Any]:
        """Multiple agents collaborate on a task"""
        pass

