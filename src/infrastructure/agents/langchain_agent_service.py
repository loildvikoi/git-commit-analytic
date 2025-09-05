# src/infrastructure/agents/langchain_agent_service.py
from typing import List, Dict, Any, Optional
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationSummaryMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import logging
import asyncio

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...domain.services.agent_service import IAgentService
from ...domain.entities.agent import Agent, AgentRole, AgentDecision, AgentCapability

logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    query: str = Field(..., description="Plain text search query")


class LangChainAgentService(IAgentService):
    """LangChain implementation of agent service using Ollama"""

    def __init__(
            self,
            ollama_url: str = "http://localhost:11434",
            default_model: str = "llama3.2:1b"
    ):
        self.ollama_url = ollama_url
        self.default_model = default_model
        self.agents: Dict[str, Agent] = {}
        self.executors: Dict[str, AgentExecutor] = {}

    async def create_agent(
            self,
            name: str,
            role: AgentRole,
            model: str = None
    ) -> Agent:
        """Create a new agent with LangChain"""

        # Create domain agent
        capabilities = self._get_capabilities_for_role(role)
        agent = Agent(
            name=name,
            role=role,
            capabilities=capabilities,
            model=model or self.default_model
        )

        # Create LangChain components
        llm = ChatOllama(
            base_url=self.ollama_url,
            model=agent.model,
            temperature=agent.temperature
        )

        # Create tools based on role
        tools = self._create_tools_for_role(role)

        # Create prompt
        prompt = self._create_prompt_for_role(role)

        # Create agent executor
        from langchain.agents import create_react_agent
        # langchain_agent = create_react_agent(
        #     llm=llm,
        #     tools=tools,
        #     prompt=prompt
        # )
        langchain_agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

        # executor = AgentExecutor(
        #     agent=langchain_agent,
        #     tools=tools,
        #     verbose=True,
        #     max_iterations=6,
        #     return_intermediate_steps=True,
        #     early_stopping_method="generate",
        #     handle_parsing_errors=False,
        # )
        executor = AgentExecutor(
            agent=langchain_agent,
            tools=tools,
            verbose=True,
            max_iterations=6,
            return_intermediate_steps=True,
        )

        # Store agent and executor
        self.agents[agent.id] = agent
        self.executors[agent.id] = executor

        logger.info(f"Created agent: {name} with role {role.value}")
        return agent

    async def execute_task(
            self,
            agent: Agent,
            task: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task with an agent"""

        executor = self.executors.get(agent.id)
        if not executor:
            raise ValueError(f"Agent {agent.id} not found")

        # Update agent status
        agent.status = "working"
        agent.current_task = task

        try:
            # Prepare input with context
            input_text = self._prepare_input(task, context)

            # Execute with LangChain
            result = await asyncio.to_thread(
                executor.invoke,
                {"input": input_text}
            )

            # Extract output
            output = result.get("output", "")

            # Update agent
            agent.status = "idle"
            agent.current_task = None

            return {
                "success": True,
                "output": output,
                "intermediate_steps": result.get("intermediate_steps", [])
            }

        except Exception as e:
            logger.error(f"Agent task execution failed: {str(e)}")
            agent.status = "error"
            return {
                "success": False,
                "error": str(e),
                "output": None
            }

    async def get_agent_decision(
            self,
            agent: Agent,
            situation: str,
            options: List[str]
    ) -> AgentDecision:
        """Get agent's decision for a situation"""

        # Create decision prompt
        prompt = f"""
        Situation: {situation}

        Available options:
        {chr(10).join(f"{i + 1}. {opt}" for i, opt in enumerate(options))}

        Analyze the situation and choose the best option.
        Explain your reasoning step by step.

        Format your response as:
        DECISION: [chosen option]
        REASONING: [your explanation]
        CONFIDENCE: [0.0 to 1.0]
        """

        # Execute decision task
        result = await self.execute_task(agent, prompt)

        # Parse response
        output = result.get("output", "")
        decision = self._parse_decision(output, options)

        # Create decision object
        agent_decision = AgentDecision(
            action=decision["action"],
            reasoning=decision["reasoning"],
            confidence=decision["confidence"],
            parameters={"situation": situation, "options": options}
        )

        # Record decision
        agent.add_decision(agent_decision)

        return agent_decision

    async def collaborate(
            self,
            agents: List[Agent],
            task: str
    ) -> Dict[str, Any]:
        """Multiple agents collaborate on a task"""

        results = {}
        context = {}

        # Sequential collaboration (can be made parallel)
        for agent in agents:
            # Each agent works with context from previous agents
            agent_result = await self.execute_task(
                agent=agent,
                task=task,
                context=context
            )

            results[agent.name] = agent_result

            # Update context for next agent
            context[f"{agent.name}_output"] = agent_result.get("output", "")

        # Synthesize results
        synthesis = self._synthesize_collaboration(results)

        return {
            "task": task,
            "agent_results": results,
            "synthesis": synthesis
        }

    def _get_capabilities_for_role(self, role: AgentRole) -> List[AgentCapability]:
        """Get capabilities based on role"""
        role_capabilities = {
            AgentRole.CODE_REVIEWER: [
                AgentCapability.ANALYZE_CODE,
                AgentCapability.SEARCH_COMMITS,
                AgentCapability.GENERATE_REPORT
            ],
            AgentRole.BUG_DETECTOR: [
                AgentCapability.ANALYZE_CODE,
                AgentCapability.SEARCH_COMMITS,
                AgentCapability.MAKE_DECISIONS
            ],
            AgentRole.PERFORMANCE_ANALYZER: [
                AgentCapability.ANALYZE_CODE,
                AgentCapability.GENERATE_REPORT
            ],
            AgentRole.SECURITY_SCANNER: [
                AgentCapability.ANALYZE_CODE,
                AgentCapability.MAKE_DECISIONS
            ],
            AgentRole.DOCUMENTATION_WRITER: [
                AgentCapability.GENERATE_REPORT,
                AgentCapability.ANSWER_QUESTIONS
            ],
            AgentRole.GENERAL_ASSISTANT: [
                AgentCapability.ANSWER_QUESTIONS,
                AgentCapability.SEARCH_COMMITS,
                AgentCapability.EXECUTE_TOOLS
            ]
        }
        return role_capabilities.get(role, [AgentCapability.ANSWER_QUESTIONS])

    def _create_tools_for_role(self, role: AgentRole) -> List[Tool]:
        """Create LangChain tools based on role"""

        # Common tools for all agents
        common_tools = [
            StructuredTool.from_function(
                func=self._search_commits_tool,
                name="search_commits",
                description="Search relevant commits.",
                args_schema=SearchInput,
            ),
            StructuredTool.from_function(
                func=self._analyze_code_tool,
                name="analyze_code",
                description="Analyze code quality.",
            ),
        ]

        # Role-specific tools
        # if role == AgentRole.CODE_REVIEWER:
        #     common_tools.extend([
        #         Tool(
        #             name="check_style",
        #             func=self._check_style_tool,
        #             description="Check code style compliance"
        #         ),
        #         Tool(
        #             name="suggest_improvements",
        #             func=self._suggest_improvements_tool,
        #             description="Suggest code improvements"
        #         )
        #     ])
        # elif role == AgentRole.BUG_DETECTOR:
        #     common_tools.extend([
        #         Tool(
        #             name="detect_patterns",
        #             func=self._detect_patterns_tool,
        #             description="Detect bug patterns in code"
        #         )
        #     ])

        return common_tools

    def _create_prompt_for_role(self, role: AgentRole) -> ChatPromptTemplate:
        role_prompts = {
            AgentRole.CODE_REVIEWER: "You are an expert Code Reviewer.",
            AgentRole.BUG_DETECTOR: "You are a Bug Detection Specialist.",
            AgentRole.PERFORMANCE_ANALYZER: "You are a Performance Analysis Expert.",
            AgentRole.SECURITY_SCANNER: "You are a Security Expert.",
            AgentRole.DOCUMENTATION_WRITER: "You are a Technical Documentation Expert.",
            AgentRole.GENERAL_ASSISTANT: "You are a helpful AI assistant for developers."
        }
        base_prompt = role_prompts.get(role, role_prompts[AgentRole.GENERAL_ASSISTANT])

        system_instructions = f"""{base_prompt}

    You have access to the following tools:

    {{tools}}

    HARD RULES (follow EXACTLY, no exceptions):
    - When you call a tool, output EXACTLY two lines, no markdown, no backticks, no asterisks:
      Action: one of [{{tool_names}}]
      Action Input: <plain text only>
    - Do NOT write "Input:". It MUST be "Action Input:".
    - After outputting Action + Action Input, STOP and wait.
    - When done:
      Thought: I now know the final answer
      Final Answer: <answer>
    """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        return prompt

    def _prepare_input(self, task: str, context: Optional[Dict]) -> str:
        """Prepare input text with context"""
        if not context:
            return task

        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        return f"{task}\n\nContext:\n{context_str}"

    def _parse_decision(self, output: str, options: List[str]) -> Dict[str, Any]:
        """Parse decision from agent output"""

        # Default values
        decision = {
            "action": options[0] if options else "UNKNOWN",
            "reasoning": "Unable to parse reasoning",
            "confidence": 0.5
        }

        # Try to parse structured output
        lines = output.split('\n')
        for line in lines:
            if line.startswith("DECISION:"):
                decision["action"] = line.replace("DECISION:", "").strip()
            elif line.startswith("REASONING:"):
                decision["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                    decision["confidence"] = max(0, min(1, confidence))
                except:
                    pass

        return decision

    def _synthesize_collaboration(self, results: Dict[str, Any]) -> str:
        """Synthesize results from multiple agents"""
        synthesis = "Collaboration Summary:\n\n"

        for agent_name, result in results.items():
            output = result.get("output", "No output")
            synthesis += f"{agent_name}:\n{output[:200]}...\n\n"

        return synthesis

    # Tool implementations
    def _search_commits_tool(self, query: str) -> str:
        """Tool to search commits"""
        # This would integrate with your search service
        return f"Searched for: {query}. Found 5 relevant commits."

    def _analyze_code_tool(self, code: str) -> str:
        """Tool to analyze code"""
        return f"Code analysis complete. No critical issues found."

    def _check_style_tool(self, code: str) -> str:
        """Tool to check code style"""
        return "Code style is compliant with standards."

    def _suggest_improvements_tool(self, code: str) -> str:
        """Tool to suggest improvements"""
        return "Consider using more descriptive variable names."

    def _detect_patterns_tool(self, code: str) -> str:
        """Tool to detect bug patterns"""
        return "No common bug patterns detected."