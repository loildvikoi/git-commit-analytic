# src/infrastructure/persistence/repositories/sqlite_agent_repository.py
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import json

from ....domain.repositories.agent_repository import IAgentRepository
from ....domain.entities.agent import Agent, AgentRole, AgentCapability, AgentDecision
from ..models.agent_model import AgentModel

logger = logging.getLogger(__name__)


class SqliteAgentRepository(IAgentRepository):
    """SQLite implementation of agent repository"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, agent: Agent) -> Agent:
        """Save an agent"""
        try:
            db_agent = AgentModel(
                id=agent.id,
                name=agent.name,
                role=agent.role.value,
                model=agent.model,
                temperature=agent.temperature,
                max_iterations=agent.max_iterations,
                status=agent.status,
                current_task=agent.current_task,
                capabilities=[c.value for c in agent.capabilities],
                tools_used=agent.tools_used,
                decisions=[self._decision_to_dict(d) for d in agent.decisions],
                created_at=agent.created_at,
                updated_at=agent.updated_at
            )

            self.session.add(db_agent)
            await self.session.commit()
            await self.session.refresh(db_agent)

            logger.info(f"Saved agent: {agent.name}")
            return self._to_domain(db_agent)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error saving agent {agent.name}: {str(e)}")
            raise

    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        try:
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.session.execute(stmt)
            db_agent = result.scalar_one_or_none()

            return self._to_domain(db_agent) if db_agent else None

        except Exception as e:
            logger.error(f"Error finding agent by ID {agent_id}: {str(e)}")
            return None

    async def find_by_role(self, role: AgentRole) -> List[Agent]:
        """Find agents by role"""
        try:
            stmt = select(AgentModel).where(AgentModel.role == role.value)
            result = await self.session.execute(stmt)
            db_agents = result.scalars().all()

            return [self._to_domain(db_agent) for db_agent in db_agents]

        except Exception as e:
            logger.error(f"Error finding agents by role {role}: {str(e)}")
            return []

    async def find_by_name(self, name: str) -> Optional[Agent]:
        """Find agent by name"""
        try:
            stmt = select(AgentModel).where(AgentModel.name == name)
            result = await self.session.execute(stmt)
            db_agent = result.scalar_one_or_none()

            return self._to_domain(db_agent) if db_agent else None

        except Exception as e:
            logger.error(f"Error finding agent by name {name}: {str(e)}")
            return None

    async def update(self, agent: Agent) -> Agent:
        """Update agent"""
        try:
            stmt = select(AgentModel).where(AgentModel.id == agent.id)
            result = await self.session.execute(stmt)
            db_agent = result.scalar_one_or_none()

            if not db_agent:
                raise ValueError(f"Agent not found: {agent.id}")

            # Update fields
            db_agent.status = agent.status
            db_agent.current_task = agent.current_task
            db_agent.tools_used = agent.tools_used
            db_agent.decisions = [self._decision_to_dict(d) for d in agent.decisions]
            db_agent.updated_at = agent.updated_at

            await self.session.commit()
            await self.session.refresh(db_agent)

            logger.info(f"Updated agent: {agent.name}")
            return self._to_domain(db_agent)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating agent {agent.id}: {str(e)}")
            raise

    async def delete(self, agent_id: str) -> bool:
        """Delete agent"""
        try:
            stmt = select(AgentModel).where(AgentModel.id == agent_id)
            result = await self.session.execute(stmt)
            db_agent = result.scalar_one_or_none()

            if not db_agent:
                return False

            await self.session.delete(db_agent)
            await self.session.commit()

            logger.info(f"Deleted agent: {agent_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting agent {agent_id}: {str(e)}")
            return False

    async def list_active_agents(self) -> List[Agent]:
        """List all active agents"""
        try:
            stmt = select(AgentModel).where(AgentModel.status != "disabled")
            result = await self.session.execute(stmt)
            db_agents = result.scalars().all()

            return [self._to_domain(db_agent) for db_agent in db_agents]

        except Exception as e:
            logger.error(f"Error listing active agents: {str(e)}")
            return []

    def _to_domain(self, db_model: AgentModel) -> Agent:
        """Convert database model to domain entity"""
        agent = Agent(
            name=db_model.name,
            role=AgentRole[db_model.role.upper()],
            capabilities=[
                AgentCapability[c.upper()]
                for c in (db_model.capabilities or [])
            ],
            model=db_model.model,
            temperature=db_model.temperature,
            max_iterations=db_model.max_iterations
        )

        # Set ID and timestamps
        agent.id = db_model.id
        agent.created_at = db_model.created_at
        agent.updated_at = db_model.updated_at

        # Set runtime state
        agent.status = db_model.status
        agent.current_task = db_model.current_task
        agent.tools_used = db_model.tools_used or []

        # Reconstruct decisions
        if db_model.decisions:
            for decision_dict in db_model.decisions:
                agent.decisions.append(
                    AgentDecision(
                        action=decision_dict["action"],
                        reasoning=decision_dict["reasoning"],
                        confidence=decision_dict["confidence"],
                        parameters=decision_dict.get("parameters", {})
                    )
                )

        return agent

    def _decision_to_dict(self, decision: AgentDecision) -> Dict:
        """Convert decision to dict for storage"""
        return {
            "action": decision.action,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "parameters": decision.parameters
        }
