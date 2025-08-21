import logging

from fastapi import APIRouter, Depends, HTTPException
from ....application.use_cases.chat_with_ai import ChatWithAIUseCase
from ....interface.dto.commit_dto import ChatRequestDto, ChatResponseDto
from ..dependencies import get_chat_use_case

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponseDto)
async def chat_with_ai(
        request: ChatRequestDto,
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
):
    """Chat with AI about commits"""

    try:
        response = await use_case.execute(request)
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/health")
async def chat_health_check(
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
):
    """Check AI service health"""

    try:
        health = await use_case.ai_analyzer.health_check()
        model_info = use_case.ai_analyzer.get_model_info()

        return {
            "status": "healthy" if health else "unhealthy",
            "model": model_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
