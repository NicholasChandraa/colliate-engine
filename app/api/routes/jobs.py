from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db_session
from app.models.job import Job, JobStatus
from app.schemas.job import JobStatusResponse, JobShotSchema
from app.services.job_service import get_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> JobStatusResponse:
    """
    Poll this endpoint to track video generation progress.
    """
    job: Job | None = await get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        product_research=job.product_research,
        progress_message=job.progress_message,
        total_shots=job.total_shots,
        completed_shots=job.completed_shots,
        shots=[JobShotSchema.model_validate(s) for s in job.shots],
        final_video_path=job.final_video_path,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> JobStatusResponse:
    """
    Cancel an ongoing job by updating its status to CANCELLED in the database.
    The ARQ worker will detect this status during its loop and gracefully abort.
    """
    from app.services.job_service import update_job_status
    import logging
    
    logger = logging.getLogger("colliate")
    
    job: Job | None = await get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    if job.status in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job because it is already {job.status.value}.")

    await update_job_status(db, job_id, status=JobStatus.CANCELLED, progress_message="Cancelled by user.")
    await db.commit()

    logger.warning(f"🛑 Received Cancel Request -> Updated DB Status to CANCELLED [Job: {job_id}]")

    # Fetch updated job
    updated_job = await get_job(db, job_id)
    if not updated_job:
        raise HTTPException(status_code=404, detail="Job not found after updating.")
    return JobStatusResponse(
        job_id=updated_job.id,
        status=updated_job.status,
        product_research=updated_job.product_research,
        progress_message=updated_job.progress_message,
        total_shots=updated_job.total_shots,
        completed_shots=updated_job.completed_shots,
        shots=[JobShotSchema.model_validate(s) for s in updated_job.shots],
        final_video_path=updated_job.final_video_path,
        error_message=updated_job.error_message,
        created_at=updated_job.created_at,
        updated_at=updated_job.updated_at,
        completed_at=updated_job.completed_at,
    )