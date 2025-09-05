# src/domain/services/tool_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable


class Tool:
    """A tool that agents can use"""

    def __init__(
            self,
            name: str,
            description: str,
            function: Callable,
            parameters: Dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters

    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return self.function(**kwargs)


class IToolService(ABC):
    """Service interface for tool management"""

    @abstractmethod
    def register_tool(self, tool: Tool) -> None:
        """Register a tool"""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        pass

    @abstractmethod
    def list_tools(self) -> List[Tool]:
        """List all available tools"""
        pass

    @abstractmethod
    async def execute_tool(
            self,
            tool_name: str,
            parameters: Dict[str, Any]
    ) -> Any:
        """Execute a tool with parameters"""
        pass
