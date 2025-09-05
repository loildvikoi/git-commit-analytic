# src/domain/repositories/agent_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.agent import Agent, AgentRole


class IAgentRepository(ABC):
    """Repository interface for Agent entity"""

    @abstractmethod
    async def save(self, agent: Agent) -> Agent:
        """Save an agent"""
        pass

    @abstractmethod
    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        pass

    @abstractmethod
    async def find_by_role(self, role: AgentRole) -> List[Agent]:
        """Find agents by role"""
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Agent]:
        """Find agent by name"""
        pass

    @abstractmethod
    async def update(self, agent: Agent) -> Agent:
        """Update agent"""
        pass

    @abstractmethod
    async def delete(self, agent_id: str) -> bool:
        """Delete agent"""
        pass

    @abstractmethod
    async def list_active_agents(self) -> List[Agent]:
        """List all active agents"""
        pass