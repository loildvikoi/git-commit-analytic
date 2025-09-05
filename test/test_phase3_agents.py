# test/test_phase3_agents.py
import asyncio
import httpx

BASE_URL = "http://localhost:8000"


async def test_create_agent():
    """Test agent creation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/agents/create",
            json={
                "name": "TestReviewer",
                "role": "code_reviewer",
                "model": "llama3.2:1b",
                "temperature": 0.3
            }
        )
        print(f"Agent created: {response.json()}")
        return response.json()["agent_id"]


async def test_code_review():
    """Test autonomous code review"""
    async with httpx.AsyncClient() as client:
        # First, get a commit ID (assuming you have commits in DB)
        commits_response = await client.get(
            f"{BASE_URL}/api/v1/commits/",
            params={"limit": 1}
        )

        if commits_response.json():
            commit_id = commits_response.json()[0]["id"]

            # Request code review
            response = await client.post(
                f"{BASE_URL}/api/v1/agents/code-review",
                json={
                    "commit_id": commit_id,
                    "review_depth": "standard"
                }
            )

            print(f"Code review result: {response.json()}")
            return response.json()


async def test_proactive_insights():
    """Test proactive insights generation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/agents/proactive-insights",
            json={
                "lookback_days": 7,
                "insight_types": ["patterns", "risks", "recommendations"]
            }
        )

        print(f"Insights: {response.json()}")
        return response.json()


async def test_multi_agent():
    """Test multi-agent collaboration"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/agents/multi-agent-analysis",
            json={
                "project": "my-project",
                "analysis_type": "comprehensive"
            }
        )

        print(f"Multi-agent analysis: {response.json()}")
        return response.json()


async def main():
    print("🚀 Testing Phase 3 Agent System\n")

    # Test 1: Create agent
    print("1. Creating agent...")
    agent_id = await test_create_agent()

    # Test 2: Code review
    print("\n2. Testing code review...")
    await test_code_review()

    # Test 3: Proactive insights
    print("\n3. Testing proactive insights...")
    await test_proactive_insights()

    # Test 4: Multi-agent analysis
    print("\n4. Testing multi-agent analysis...")
    await test_multi_agent()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
