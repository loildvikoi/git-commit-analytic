# test/health_check.py
import asyncio
import httpx


async def check_all_systems():
    """Complete system health check"""

    checks = {
        "Phase 1 - Webhook": "/api/v1/webhooks/github",
        "Phase 1 - Commits": "/api/v1/commits/",
        "Phase 2 - Documents": "/api/v1/documents/search",
        "Phase 2 - RAG": "/api/v1/rag/health",
        "Phase 3 - Agents": "/api/v1/agents/health",
        "Ollama": "http://localhost:11434/api/tags"
    }

    print("🏥 System Health Check\n")

    async with httpx.AsyncClient() as client:
        for name, endpoint in checks.items():
            try:
                if "localhost:11434" in endpoint:
                    response = await client.get(endpoint)
                else:
                    response = await client.get(f"http://localhost:8000{endpoint}")

                status = "✅" if response.status_code == 200 else "❌"
                print(f"{status} {name}: {response.status_code}")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")

    print("\n✅ Health check complete!")


if __name__ == "__main__":
    asyncio.run(check_all_systems())