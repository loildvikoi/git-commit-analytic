# test/test_phase2_rag.py
"""
Test script for Phase 2 RAG functionality
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"


async def test_document_indexing():
    """Test indexing a document"""
    print("\n📝 Testing Document Indexing...")

    async with httpx.AsyncClient() as client:
        # Index a test document
        response = await client.post(
            f"{BASE_URL}/api/v1/documents/index",
            json={
                "content": """
                This is a test commit for the RAG system.
                Author: John Doe
                Project: TestProject

                Changes made:
                - Added new authentication system using JWT tokens
                - Fixed bug in user registration flow
                - Updated database schema for better performance
                - Added unit tests for auth module

                This commit implements a complete authentication system with
                secure password hashing using bcrypt and JWT token generation
                for API authentication.
                """,
                "document_type": "commit",
                "source_type": "test",
                "project": "TestProject",
                "author": "john.doe@example.com",
                "title": "Implement authentication system",
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document indexed successfully!")
            print(f"   - Documents created: {len(result)}")
            print(f"   - First document ID: {result[0]['id']}")
            return result[0]['id']
        else:
            print(f"❌ Failed to index document: {response.status_code}")
            print(response.text)
            return None


async def test_document_search():
    """Test searching documents"""
    print("\n🔍 Testing Document Search...")

    async with httpx.AsyncClient() as client:
        # Search for documents
        response = await client.post(
            f"{BASE_URL}/api/v1/documents/search",
            json={
                "query": "authentication JWT",
                "use_hybrid": True,
                "use_cache": False,
                "max_results": 5
            }
        )

        if response.status_code == 200:
            results = response.json()
            print(f"✅ Search completed successfully!")
            print(f"   - Results found: {len(results)}")

            for i, result in enumerate(results[:3], 1):
                print(f"\n   Result {i}:")
                print(f"   - Type: {result['document']['document_type']}")
                print(f"   - Project: {result['document']['project']}")
                print(f"   - Score: {result['score']['combined']:.3f}")
                print(f"   - Title: {result['document']['title'][:50]}...")
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(response.text)


async def test_sync_commits():
    """Test syncing existing commits to documents"""
    print("\n🔄 Testing Commit Sync...")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/documents/sync-commits",
            json={
                "limit": 10,
                "skip_existing": True
            }
        )

        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            print(f"✅ Sync completed successfully!")
            print(f"   - Total commits: {data.get('total_commits', 0)}")
            print(f"   - Synced: {data.get('synced', 0)}")
            print(f"   - Skipped: {data.get('skipped', 0)}")
            print(f"   - Failed: {data.get('failed', 0)}")
        else:
            print(f"❌ Sync failed: {response.status_code}")
            print(response.text)


async def test_rag_chat():
    """Test RAG chat functionality"""
    print("\n💬 Testing RAG Chat...")

    questions = [
        "What authentication system was implemented?",
        "What recent bugs were fixed?",
        "Tell me about the database changes",
        "Who worked on the authentication features?"
    ]

    async with httpx.AsyncClient() as client:
        for question in questions:
            print(f"\n   Q: {question}")

            response = await client.post(
                f"{BASE_URL}/api/v1/rag/chat",
                json={
                    "question": question,
                    "context_project": "TestProject",
                    "max_documents": 5,
                    "use_cache": False
                }
            )

            if response.status_code == 200:
                result = response.json()
                print(f"   A: {result['answer'][:200]}...")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Sources used: {result['context_used']}")
            else:
                print(f"   ❌ Failed: {response.status_code}")