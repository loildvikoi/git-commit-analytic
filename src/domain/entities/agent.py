# src/domain/entities/agent.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from .base import Entity, ValueObject


class AgentRole(Enum):
    """Types of agents in the system"""
    CODE_REVIEWER = "code_reviewer"
    BUG_DETECTOR = "bug_detector"
    PERFORMANCE_ANALYZER = "performance_analyzer"
    SECURITY_SCANNER = "security_scanner"
    DOCUMENTATION_WRITER = "documentation_writer"
    GENERAL_ASSISTANT = "general_assistant"


class AgentCapability(Enum):
    """What agents can do"""
    ANALYZE_CODE = "analyze_code"
    SEARCH_COMMITS = "search_commits"
    GENERATE_REPORT = "generate_report"
    ANSWER_QUESTIONS = "answer_questions"
    EXECUTE_TOOLS = "execute_tools"
    MAKE_DECISIONS = "make_decisions"


@dataclass
class AgentMemory(ValueObject):
    """Agent's conversation memory"""
    conversation_id: str
    messages: List[Dict[str, str]]
    context: Dict[str, Any]
    created_at: datetime

    def _validate(self):
        if not self.conversation_id:
            raise ValueError("Conversation ID is required")
        if len(self.messages) > 100:
            raise ValueError("Memory overflow - too many messages")


@dataclass
class AgentDecision(ValueObject):
    """A decision made by an agent"""
    action: str
    reasoning: str
    confidence: float
    parameters: Dict[str, Any]

    def _validate(self):
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        if not self.action:
            raise ValueError("Action is required")


class Agent(Entity):
    """Agent entity - an autonomous AI worker"""

    def __init__(
            self,
            name: str,
            role: AgentRole,
            capabilities: List[AgentCapability],
            model: str = "llama3.2:1b",  # Default Ollama model
            temperature: float = 0.3,
            max_iterations: int = 5
    ):
        super().__init__()
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Runtime state
        self.memory: Optional[AgentMemory] = None
        self.current_task: Optional[str] = None
        self.decisions: List[AgentDecision] = []
        self.tools_used: List[str] = []
        self.status: str = "idle"

    def can_perform(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability"""
        return capability in self.capabilities

    def add_decision(self, decision: AgentDecision):
        """Record a decision made by the agent"""
        self.decisions.append(decision)
        self.updated_at = datetime.now()

    def set_memory(self, memory: AgentMemory):
        """Set agent's conversation memory"""
        self.memory = memory

    def clear_memory(self):
        """Clear agent's memory"""
        self.memory = None
        self.decisions = []
        self.tools_used = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "capabilities": [c.value for c in self.capabilities],
            "model": self.model,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "current_task": self.current_task,
            "decisions_count": len(self.decisions),
            "tools_used": self.tools_used,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
