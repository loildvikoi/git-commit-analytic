from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime

from ....application.use_cases.search_commits import SearchCommitsUseCase
from ....application.dto.commit_dto import CommitSearchDto, CommitResponseDto
from ..dependencies import get_search_commits_use_case, get_commit_repository
from ....domain.repositories.commit_repository import ICommitRepository

router = APIRouter(prefix="/commits", tags=["commits"])


@router.get("/", response_model=List[CommitResponseDto])
async def search_commits(
        q: Optional[str] = Query(None, description="Search query"),
        project: Optional[str] = Query(None, description="Filter by project"),
        author: Optional[str] = Query(None, description="Filter by author email"),
        branch: Optional[str] = Query(None, description="Filter by branch"),
        start_date: Optional[datetime] = Query(None, description="Start date filter"),
        end_date: Optional[datetime] = Query(None, description="End date filter"),
        tags: List[str] = Query([], description="Filter by tags"),
        limit: int = Query(50, ge=1, le=200, description="Result limit"),
        offset: int = Query(0, ge=0, description="Result offset"),
        use_case: SearchCommitsUseCase = Depends(get_search_commits_use_case)
):
    """Search commits with various filters"""

    search_dto = CommitSearchDto(
        query=q,
        project=project,
        author=author,
        branch=branch,
        start_date=start_date,
        end_date=end_date,
        tags=tags,
        limit=limit,
        offset=offset
    )

    results = await use_case.execute(search_dto)
    return results


@router.get("/{commit_id}", response_model=CommitResponseDto)
async def get_commit(
        commit_id: str,
        repo: ICommitRepository = Depends(get_commit_repository)
):
    """Get specific commit by ID"""

    commit = await repo.find_by_id(commit_id)
    if not commit:
        raise HTTPException(status_code=404, detail="Commit not found")

    # Convert to response DTO
    from ....application.use_cases.search_commits import SearchCommitsUseCase
    search_use_case = SearchCommitsUseCase(repo, None)
    return search_use_case._to_response_dto(commit)


@router.get("/stats/overview")
async def get_commit_statistics(
        project: Optional[str] = Query(None),
        author: Optional[str] = Query(None),
        start_date: Optional[datetime] = Query(None),
        end_date: Optional[datetime] = Query(None),
        repo: ICommitRepository = Depends(get_commit_repository)
):
    """Get commit statistics"""

    stats = await repo.get_statistics(
        project=project,
        author=author,
        start_date=start_date,
        end_date=end_date
    )

    return {
        "status": "success",
        "data": stats
    }
