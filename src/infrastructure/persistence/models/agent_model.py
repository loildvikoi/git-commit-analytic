# src/infrastructure/persistence/models/agent_model.py
from sqlalchemy import Column, String, Float, Integer, JSON, DateTime, Text
from sqlalchemy.sql import func
from ..database import Base


class AgentModel(Base):
    """SQLAlchemy model for agents"""

    __tablename__ = "agents"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    role = Column(String(50), nullable=False, index=True)
    model = Column(String(100))
    temperature = Column(Float, default=0.3)
    max_iterations = Column(Integer, default=5)
    status = Column(String(20), default="idle")
    current_task = Column(Text)
    capabilities = Column(JSON, default=list)
    tools_used = Column(JSON, default=list)
    decisions = Column(JSON, default=list)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<AgentModel(name={self.name}, role={self.role})>"


class WorkflowModel(Base):
    """SQLAlchemy model for workflows"""

    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    steps = Column(JSON, nullable=False)
    parallel_execution = Column(Integer, default=0)  # Boolean as int
    current_step_index = Column(Integer, default=0)
    status = Column(String(20), default="pending")
    results = Column(JSON, default=dict)
    errors = Column(JSON, default=list)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<WorkflowModel(name={self.name}, status={self.status})>"
