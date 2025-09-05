# Phase 3 Setup Guide - Agent System

## 📋 Prerequisites

- Phase 2 working correctly
- Ollama running with llama3.2:1b model
- Python 3.10+

## 📊 Key Concepts to Learn

### 1. **Agent** 
- Autonomous decision-making entity
- Has memory, tools, and reasoning capability
- Can plan and execute multi-step tasks

### 2. **LangChain**
- Most popular framework for LLM applications
- Provides abstractions for agents, tools, memory
- Integrates with many LLMs (including Ollama)

### 3. **ReAct Pattern**
- Reasoning + Acting
- Agent thinks step-by-step
- Format: Thought → Action → Observation → Repeat

### 4. **Tools**
- Functions agents can call
- Examples: search, calculate, analyze
- Extend agent capabilities

### 5. **Memory**
- Conversation history
- Context preservation
- Different types: buffer, summary, vector

### 6. **Workflow Orchestration**
- Coordinate multiple agents
- Sequential or parallel execution
- State management between steps

## 🧩 Architecture Overview

```plaintext
┌─────────────────────────────────────────────────────────┐
│                     PHASE 3: AGENTS                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │ • Autonomous Agents  • Multi-step Reasoning     │    │
│  │ • Tool Usage        • Workflow Orchestration    │    │
│  └─────────────────────────────────────────────────┘    │
│                           ↓ Uses                        │
├─────────────────────────────────────────────────────────┤
│                    PHASE 2: RAG                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ • Vector Search     • Semantic Understanding    │    │
│  │ • Embeddings        • Hybrid Search             │    │
│  └─────────────────────────────────────────────────┘    │
│                           ↓ Enhances                    │
├─────────────────────────────────────────────────────────┤
│                   PHASE 1: BASIC AI                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │ • Webhook Input     • Commit Storage            │    │
│  │ • Ollama Analysis   • Basic Search              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Learning Path

1. **Start Simple**
   - Single agent with basic tools
   - Simple prompts and tasks
   - Understand ReAct loop

2. **Add Complexity**
   - Multiple agents
   - Custom tools
   - Memory management

3. **Advanced Features**
   - Agent collaboration
   - Workflow orchestration
   - Predictive analytics

## 🐛 Common Issues & Solutions

### Issue: "Ollama not responding"
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Issue: "Agent stuck in loop"
```python
# Reduce max_iterations
agent = Agent(
    max_iterations=3  # Lower from 5
)
```

### Issue: "Poor agent responses"
```python
# Improve prompts
# Be more specific about format
# Provide examples
# Use lower temperature for consistency
```

### Issue: "Tools not working"
```python
# Check tool descriptions are clear
# Ensure tool functions return strings
# Add error handling in tools
```

## 📈 Monitoring & Debugging

### Enable LangChain Tracing

```python
# For debugging
import langchain
langchain.debug = True

# Or use LangSmith (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
```

### Log Agent Decisions

```python
# Track agent reasoning
logger.info(f"Agent decision: {agent_decision.action}")
logger.info(f"Reasoning: {agent_decision.reasoning}")
logger.info(f"Confidence: {agent_decision.confidence}")
```

## 🚀 Next Steps

1. **Experiment with Different Models**
   ```bash
   # Try other Ollama models
   ollama pull codellama:7b
   ollama pull mistral:7b
   ```

2. **Create Custom Tools**
   ```python
   # Add domain-specific tools
   - Git operations
   - Code analysis
   - Database queries
   ```

3. **Build Complex Workflows**
   ```python
   # Multi-stage analysis
   # Automated PR reviews
   # Daily insights reports
   ```

4. **Add Evaluation**
   ```python
   # Measure agent performance
   # A/B testing different prompts
   # User feedback integration
   ```

## 📚 Resources

- **LangChain Docs**: https://python.langchain.com/docs/
- **Ollama Models**: https://ollama.ai/library
- **Agent Architectures**: Search "ReAct", "Chain of Thought", "Tree of Thoughts"
- **Prompt Engineering**: Search "few-shot", "chain-of-thought", "self-consistency"
