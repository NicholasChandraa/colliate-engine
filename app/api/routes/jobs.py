import traceback
import uuid
from typing import Annotated
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete as sa_delete
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.logging import logger
from app.models.job import Job, JobShot, JobStatus
from app.schemas.job import JobStatusResponse, JobShotSchema, SelectImageRequest, ApproveJobResponse
from app.services.job_service import get_job, update_job_status, select_shot_image, get_selected_shots, get_user_jobs, create_job
from app.api.middleware.auth import require_user_id

router = APIRouter(prefix="/jobs", tags=["jobs"])

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        title=job.title,
        status=job.status,
        product_name=job.product_name,
        product_research=job.product_research,
        progress_message=job.progress_message,
        total_shots=job.total_shots,
        completed_shots=job.completed_shots,
        shots=[JobShotSchema.model_validate(s) for s in job.shots],
        product_image_path=job.product_image_path,
        reference_image_path=job.reference_image_path,
        reference_image_type=job.reference_image_type,
        final_video_path=job.final_video_path,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )

_TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.REJECTED}
_CANCELLABLE_STATUSES = {
    JobStatus.PENDING, JobStatus.RESEARCHING, JobStatus.DIRECTING,
    JobStatus.GENERATING_IMAGES, JobStatus.AWAITING_SELECTION,
    JobStatus.GENERATING_VIDEOS, JobStatus.ASSEMBLING,
}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[JobStatusResponse])
async def list_user_jobs(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    user_id: Annotated[str, Depends(require_user_id)],
) -> list[JobStatusResponse]:
    """Return the authenticated user's jobs (newest first, max 20)."""
    jobs = await get_user_jobs(db, user_id)
    return [_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> JobStatusResponse:
    """Poll this endpoint to track video generation progress."""
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return _to_response(job)


@router.post("/{job_id}/shots/{shot_index}/select", response_model=JobShotSchema)
async def select_shot(
    job_id: str,
    shot_index: int,
    body: SelectImageRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> JobShotSchema:
    """
    Select image option 1 or 2 for a specific shot.
    Can be called while job is AWAITING_SELECTION.
    """
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job.status != JobStatus.AWAITING_SELECTION:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot select images — job is '{job.status.value}', expected 'awaiting_selection'."
        )

    shot = await select_shot_image(db, job_id, shot_index, body.selection)
    if not shot:
        raise HTTPException(status_code=404, detail=f"Shot {shot_index} not found in job {job_id}.")

    await db.commit()
    logger.info(f"🖼️ Shot {shot_index} image {body.selection} selected [Job: {job_id}]")
    return JobShotSchema.model_validate(shot)


@router.post("/{job_id}/approve", response_model=ApproveJobResponse)
async def approve_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> ApproveJobResponse:
    """
    Approve selected images and trigger video generation (Task 2).
    At least 1 shot must be selected before approving.
    Unselected shots are skipped — only selected shots get videos generated.
    """
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job.status != JobStatus.AWAITING_SELECTION:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve — job is '{job.status.value}', expected 'awaiting_selection'."
        )

    selected_shots = await get_selected_shots(db, job_id)
    if not selected_shots:
        raise HTTPException(
            status_code=400,
            detail="No shots have been selected. Select at least 1 shot before approving."
        )

    try:
        settings = get_settings()
        redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
        await redis.enqueue_job("generate_videos_task", job_id=job_id, _job_id=f"{job_id}-videos")
        await redis.close()
    except Exception as e:
        logger.error(f"❌ Failed to enqueue generate_videos_task [Job: {job_id}] -> {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail="Service unavailable, failed to queue video generation.")

    logger.info(f"✅ Job approved — {len(selected_shots)} shots queued for video generation [Job: {job_id}]")

    return ApproveJobResponse(
        job_id=job_id,
        status="generating_videos",
        selected_shots=len(selected_shots),
        message=f"Video generation started for {len(selected_shots)} selected shots.",
    )


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> JobStatusResponse:
    """Cancel an ongoing job."""
    import logging
    _logger = logging.getLogger("colliate")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job.status in _TERMINAL_STATUSES:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job — already {job.status.value}.")

    await update_job_status(db, job_id, status=JobStatus.CANCELLED, progress_message="Cancelled by user.")
    await db.commit()

    _logger.warning(f"🛑 Job cancelled [Job: {job_id}]")

    updated_job = await get_job(db, job_id)
    if not updated_job:
        raise HTTPException(status_code=404, detail="Job not found after updating.")
    return _to_response(updated_job)
    
@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    user_id: Annotated[str, Depends(require_user_id)],
) -> None:
    """Permanently delete a job record (terminal statuses only, owner only)."""
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this job.")
    if job.status not in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job in '{job.status.value}' state. Cancel it first.",
        )

    # Delete related shots first (no FK CASCADE on job_shots)
    await db.execute(sa_delete(JobShot).where(JobShot.job_id == job_id))
    await db.execute(sa_delete(Job).where(Job.id == job_id))
    await db.commit()
    logger.info(f"\U0001f5d1\ufe0f  Job deleted [Job: {job_id}] [User: {user_id}]")


@router.post("/{job_id}/retry", response_model=JobStatusResponse)
async def retry_job(
    job_id: str,
    db: Annotated[AsyncSession, Depends(get_db_session)],
    user_id: Annotated[str, Depends(require_user_id)],
) -> JobStatusResponse:
    """Creates a new job using the product details and images of a failed/cancelled job."""
    old_job = await get_job(db, job_id)
    if not old_job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if old_job.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not allowed to retry this job.")
    if old_job.status not in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Only terminal jobs (failed/cancelled/done) can be retried.",
        )

    # Re-read the old image from disk to pass to the worker
    if not old_job.product_image_path:
        raise HTTPException(status_code=500, detail="Original product image is missing.")
    
    try:
        with open(old_job.product_image_path, "rb") as f:
            product_bytes = f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Original product image file not found on disk.")

    reference_bytes = None
    if old_job.reference_image_path:
        try:
            with open(old_job.reference_image_path, "rb") as f:
                reference_bytes = f.read()
        except FileNotFoundError:
            pass # Continue without reference if missing

    new_job_id = str(uuid.uuid4())
    logger.info(f"🔄 Retrying Job: [{job_id}] -> New Job: [{new_job_id}]")

    # Create new DB record
    await create_job(
        db=db,
        job_id=new_job_id,
        title=old_job.title,
        product_name=old_job.product_name,
        user_id=user_id,
        product_image_path=old_job.product_image_path,
        reference_image_path=old_job.reference_image_path,
        reference_image_type=old_job.reference_image_type,
    )

    # Dispatch to ARQ
    settings = get_settings()
    try:
        redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
        await redis.enqueue_job(
            "generate_images_task",
            new_job_id,
            old_job.product_name,
            product_bytes,
            reference_bytes,
            old_job.reference_image_type,
        )
        await redis.close()
    except Exception as e:
        logger.error(f"\u274c Failed to enqueue retry job to Redis: {e}\n{traceback.format_exc()}")
        await update_job_status(db, new_job_id, status=JobStatus.FAILED)
        failed_job = await get_job(db, new_job_id)
        if failed_job:
            failed_job.error_message = "Failed to enqueue retry dispatch"
            await db.commit()
        raise HTTPException(status_code=500, detail="Could not dispatch retry job to queue")

    new_job = await get_job(db, new_job_id)
    if not new_job:
         raise HTTPException(status_code=500, detail="Failed to retrieve newly created job from database")
    return _to_response(new_job)
