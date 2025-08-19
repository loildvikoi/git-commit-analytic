from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException
from typing import List
import logging

from ....application.use_cases.process_commit import ProcessCommitUseCase
from ....application.dto.webhook_dto import WebhookPayloadDto
from ....application.dto.commit_dto import CommitDto, FileChangeDto
from ..dependencies import get_process_commit_use_case, verify_webhook_signature

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/github")
async def handle_github_webhook(
        payload: WebhookPayloadDto,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
):
    """Handle GitHub webhook"""

    if not signature_verified:
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )

    logger.info(f"Received webhook for {len(payload.commits)} commits in {payload.repository.name}")

    processed_commits = []

    # Process each commit
    for commit_data in payload.commits:
        try:
            # Convert webhook data to commit DTO
            commit_dto = CommitDto(
                commit_hash=commit_data.id,
                author_email=commit_data.author.email,
                author_name=commit_data.author.name,
                message=commit_data.message,
                timestamp=commit_data.timestamp,
                branch=payload.ref.split('/')[-1],  # Extract branch from refs/heads/main
                project=payload.repository.name,
                files_changed=_convert_file_changes(commit_data),
                issue_numbers=_extract_issue_numbers(commit_data.message)
            )

            # Process commit (this will queue analysis automatically)
            result = await use_case.execute(commit_dto)
            processed_commits.append(result.commit_hash)

            logger.info(f"Processed commit: {commit_data.id[:8]}")

        except Exception as e:
            logger.error(f"Error processing commit {commit_data.id}: {str(e)}")
            # Continue processing other commits

    return {
        "status": "success",
        "message": f"Processed {len(processed_commits)} commits",
        "commits": processed_commits,
        "repository": payload.repository.name
    }


@router.post("/gitlab")
async def handle_gitlab_webhook(
        payload: dict,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
):
    """Handle GitLab webhook"""

    if not signature_verified:
        raise HTTPException(
            status_code=401,
            detail="Invalid webhook signature"
        )

    # Convert GitLab payload to our standard format
    # This is a simplified implementation
    logger.info("Received GitLab webhook (simplified handling)")

    return {
        "status": "success",
        "message": "GitLab webhook received (basic handling)",
        "commits": []
    }


def _convert_file_changes(commit_data) -> List[FileChangeDto]:
    """Convert webhook file changes to our DTO format"""
    file_changes = []

    # Added files
    for filename in commit_data.added:
        file_changes.append(FileChangeDto(
            filename=filename,
            additions=10,  # Placeholder - GitHub doesn't provide this in webhook
            deletions=0,
            status="added"
        ))

    # Modified files
    for filename in commit_data.modified:
        file_changes.append(FileChangeDto(
            filename=filename,
            additions=5,  # Placeholder
            deletions=2,  # Placeholder
            status="modified"
        ))

    # Removed files
    for filename in commit_data.removed:
        file_changes.append(FileChangeDto(
            filename=filename,
            additions=0,
            deletions=20,  # Placeholder
            status="deleted"
        ))

    return file_changes


def _extract_issue_numbers(message: str) -> List[str]:
    """Extract issue numbers from commit message"""
    import re

    # Look for patterns like #123, fixes #456, closes #789
    patterns = [
        r'#(\d+)',
        r'(?:fix|fixes|close|closes|resolve|resolves)\s+#(\d+)',
        r'(?:fix|fixes|close|closes|resolve|resolves)\s+(\d+)'
    ]

    issue_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, message, re.IGNORECASE)
        issue_numbers.extend(matches)

    return list(set(issue_numbers))  # Remove duplicates
