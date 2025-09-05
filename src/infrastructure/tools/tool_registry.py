# src/infrastructure/tools/tool_registry.py
from typing import Dict, Optional, List, Any
import logging

from ...domain.services.tool_service import IToolService, Tool

logger = logging.getLogger(__name__)


class ToolRegistry(IToolService):
    """Registry for agent tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def register_tool(self, tool: Tool) -> None:
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all available tools"""
        return list(self.tools.values())

    async def execute_tool(
            self,
            tool_name: str,
            parameters: Dict[str, Any]
    ) -> Any:
        """Execute a tool with parameters"""
        tool = self.get_tool(tool_name)

        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        try:
            result = await tool.execute(**parameters)
            logger.info(f"Executed tool: {tool_name}")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
            raise

    def _register_default_tools(self):
        """Register default tools"""

        # Search commits tool
        self.register_tool(Tool(
            name="search_commits",
            description="Search for commits in the repository",
            function=self._search_commits,
            parameters={"query": "string", "limit": "integer"}
        ))

        # Analyze code tool
        self.register_tool(Tool(
            name="analyze_code",
            description="Analyze code for quality issues",
            function=self._analyze_code,
            parameters={"code": "string"}
        ))

        # Calculate metrics tool
        self.register_tool(Tool(
            name="calculate_metrics",
            description="Calculate code metrics",
            function=self._calculate_metrics,
            parameters={"file_path": "string"}
        ))

    async def _search_commits(self, query: str, limit: int = 10) -> str:
        """Tool implementation: search commits"""
        # This would integrate with actual search
        return f"Found {limit} commits matching '{query}'"

    async def _analyze_code(self, code: str) -> str:
        """Tool implementation: analyze code"""
        lines = code.split('\n')
        return f"Analyzed {len(lines)} lines of code. No critical issues found."

    async def _calculate_metrics(self, file_path: str) -> Dict[str, Any]:
        """Tool implementation: calculate metrics"""
        return {
            "complexity": 5,
            "maintainability": 8,
            "test_coverage": 0.75
        }