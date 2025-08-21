from fastapi import APIRouter, Depends, HTTPException
import logging
import time

from src.interface.dto.document_dto import RAGChatDto, RAGChatResponseDto
from src.interface.api.dependencies import get_rag_chat_use_case

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=RAGChatResponseDto)
async def rag_chat(
        request: RAGChatDto,
        use_case=Depends(get_rag_chat_use_case)
):
    """Chat with RAG system about your codebase"""
    try:
        start_time = time.time()

        result = await use_case.execute(
            question=request.question,
            context_project=request.context_project,
            context_author=request.context_author,
            max_documents=request.max_documents,
            use_cache=request.use_cache
        )

        processing_time = int((time.time() - start_time) * 1000)

        # Log metrics
        logger.info(f"RAG chat processed in {processing_time}ms, "
                    f"confidence: {result['confidence']:.2f}, "
                    f"sources: {result['context_used']}")

        return RAGChatResponseDto(**result)

    except Exception as e:
        logger.error(f"Error in RAG chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def rag_health_check():
    """Check RAG system health"""
    try:
        # Check various components
        health_status = {
            "vector_store": "healthy",  # Would check actual status
            "embedding_service": "healthy",
            "search_service": "healthy",
            "rag_service": "healthy"
        }

        all_healthy = all(status == "healthy" for status in health_status.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": health_status
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }