from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.job import Job, JobStatus, JobShot, JobShotStatus
from app.core.logging import logger
import traceback

async def get_user_jobs(
    db: AsyncSession,
    user_id: str,
    limit: int = 20,
) -> list[Job]:
    """Return a user's jobs ordered by creation time (newest first)."""
    try:
        result = await db.execute(
            select(Job)
            .where(Job.user_id == user_id)
            .options(selectinload(Job.shots))
            .order_by(Job.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    except Exception as e:
        logger.error(f"❌ Failed to fetch jobs for user {user_id} | Error: {e}")
        raise


async def create_job(
    db: AsyncSession,
    job_id: str,
    product_name: str,
    user_id: str | None = None,
    product_image_path: str | None = None,
    reference_image_path: str | None = None,
    reference_image_type: str | None = None,
    title: str | None = None,
) -> Job:
    logger.debug(f"🗄️ Preparing DB record for new Job [{job_id}]...")
    try:
        final_title = title if title and title.strip() else product_name

        job = Job(
            id=job_id,
            title=final_title,
            product_name=product_name,
            user_id=user_id,
            product_image_path=product_image_path,
            reference_image_path=reference_image_path,
            reference_image_type=reference_image_type,
            status=JobStatus.PENDING
        )

        db.add(job)
        await db.flush()
        logger.info(f"✅ DB Record Created [Job: {job_id}]")
        return job
    except Exception as e:
        logger.error(f"❌ Failed to create DB record [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def get_job(db: AsyncSession, job_id: str) -> Job | None:
    try:
        result = await db.execute(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.shots))
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"❌ Failed to fetch job from DB [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def update_job_status(
    db: AsyncSession,
    job_id: str,
    status: JobStatus,
    progress_message: str | None = None,
    total_shots: int | None = None,
    completed_shots: int | None = None,
) -> None:

    try:
        job = await get_job(db, job_id)
        if not job:
            logger.warning(f"⚠️ Could not update DB: Job not found [Job: {job_id}]")
            return

        job.status = status
        job.updated_at = datetime.now(timezone.utc)

        if progress_message is not None:
            job.progress_message = progress_message
        if total_shots is not None:
            job.total_shots = total_shots
        if completed_shots is not None:
            job.completed_shots = completed_shots
        
        await db.flush()
        logger.debug(f"🔄 DB Status Updated [Job: {job_id} -> {status.value}]")
    except Exception as e:
        logger.error(f"❌ Failed to update DB status [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def complete_job(
    db: AsyncSession,
    job_id: str,
    final_video_path: str,
) -> None:
    try:
        job = await get_job(db, job_id)
        if not job:
            logger.warning(f"⚠️ Could not complete DB record: Job not found [Job: {job_id}]")
            return

        job.status = JobStatus.DONE
        job.final_video_path = final_video_path
        job.completed_at = datetime.now(timezone.utc)
        job.updated_at = datetime.now(timezone.utc)
        await db.flush()
        logger.info(f"🎯 DB Status Completed [Job: {job_id} -> DONE]")
    except Exception as e:
        logger.error(f"❌ Failed to mark final video in DB [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def reject_job(
    db: AsyncSession,
    job_id: str,
    reason: str,
) -> None:
    try:
        job = await get_job(db, job_id)
        if not job:
            logger.warning(f"⚠️ Could not reject DB record: Job not found [Job: {job_id}]")
            return

        job.status = JobStatus.REJECTED
        job.error_message = reason
        job.updated_at = datetime.now(timezone.utc)
        await db.flush()
        logger.warning(f"🚫 DB Status Rejected [Job: {job_id}] | Reason: {reason}")
    except Exception as e:
        logger.error(f"❌ Failed to write rejected state to DB [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def fail_job(
    db: AsyncSession,
    job_id: str,
    error_message: str,
) -> None:
    try:
        job = await get_job(db, job_id)
        if not job:
            logger.warning(f"⚠️ Could not fail DB record: Job not found [Job: {job_id}]")
            return

        job.status = JobStatus.FAILED
        job.error_message = error_message
        job.updated_at = datetime.now(timezone.utc)
        await db.flush()
        logger.warning(f"🚨 DB Status Failed [Job: {job_id}] | Fallback Reason: {error_message}")
    except Exception as e:
        logger.error(f"❌ Failed to write failure state to DB [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def create_job_shots(
    db: AsyncSession,
    job_id: str,
    storyboard_shots: list[dict[str, object]]
) -> None:
    try:
        shots = []
        for index, shot_data in enumerate(storyboard_shots, start=1):
            shot = JobShot(
                job_id=job_id,
                shot_index=index,
                shot_type=str(shot_data.get("type", "unknown")),
                camera_angle=str(shot_data.get("camera_angle")) if shot_data.get("camera_angle") else None,
                camera_movement=str(shot_data.get("camera_movement")) if shot_data.get("camera_movement") else None,
                subject_action=str(shot_data.get("subject_action")) if shot_data.get("subject_action") else None,
                lighting=str(shot_data.get("lighting")) if shot_data.get("lighting") else None,
                emotion=str(shot_data.get("emotion")) if shot_data.get("emotion") else None,
                voiceover_text=str(shot_data.get("voiceover_text")) if shot_data.get("voiceover_text") else None,
                image_prompt=str(shot_data.get("image_prompt")) if shot_data.get("image_prompt") else None,
                video_prompt=str(shot_data.get("video_prompt")) if shot_data.get("video_prompt") else None,
                negative_prompt=str(shot_data.get("negative_prompt")) if shot_data.get("negative_prompt") else None,
                status=JobShotStatus.PENDING
            )
            shots.append(shot)
            db.add(shot)

        await db.flush()
        logger.info(f"💾 Inserted {len(shots)} JobShots into DB [Job: {job_id}]")
    except Exception as e:
        logger.error(f"❌ Failed to bulk insert job shots [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def get_job_shot(db: AsyncSession, job_id: str, shot_index: int) -> JobShot | None:
    try:
        result = await db.execute(
            select(JobShot).where(JobShot.job_id == job_id, JobShot.shot_index == shot_index)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"❌ Failed to fetch job shot {shot_index} [Job: {job_id}] | Error: {e}\n{traceback.format_exc()}")
        raise


async def update_job_shot_image(
    db: AsyncSession,
    job_id: str,
    shot_index: int,
    image_path: str,
    image_number: int = 1,  # 1 or 2
) -> None:
    try:
        shot = await get_job_shot(db, job_id, shot_index)
        if shot:
            if image_number == 1:
                shot.scene_image_path = image_path
            else:
                shot.scene_image_path_2 = image_path
            shot.updated_at = datetime.now(timezone.utc)
            await db.flush()
    except Exception as e:
        logger.error(f"❌ Failed to update image {image_number} path for shot {shot_index} [Job: {job_id}] | Error: {e}")
        raise


async def select_shot_image(
    db: AsyncSession,
    job_id: str,
    shot_index: int,
    selection: int,  # 1 or 2
) -> JobShot | None:
    try:
        shot = await get_job_shot(db, job_id, shot_index)
        if not shot:
            return None
        shot.selected_image = selection
        shot.updated_at = datetime.now(timezone.utc)
        await db.flush()
        logger.info(f"✅ Shot {shot_index} image {selection} selected [Job: {job_id}]")
        return shot
    except Exception as e:
        logger.error(f"❌ Failed to select image for shot {shot_index} [Job: {job_id}] | Error: {e}")
        raise


async def get_selected_shots(db: AsyncSession, job_id: str) -> list[JobShot]:
    """Return shots that have been selected by the user, ordered by shot_index."""
    try:
        result = await db.execute(
            select(JobShot)
            .where(JobShot.job_id == job_id, JobShot.selected_image.is_not(None))
            .order_by(JobShot.shot_index)
        )
        return list(result.scalars().all())
    except Exception as e:
        logger.error(f"❌ Failed to fetch selected shots [Job: {job_id}] | Error: {e}")
        raise


async def update_job_shot_video(
    db: AsyncSession,
    job_id: str,
    shot_index: int,
    video_path: str,
) -> None:
    try:
        shot = await get_job_shot(db, job_id, shot_index)
        if shot:
            shot.video_clip_path = video_path
            shot.status = JobShotStatus.DONE
            shot.updated_at = datetime.now(timezone.utc)
            await db.flush()
    except Exception as e:
        logger.error(f"❌ Failed to update video path for shot {shot_index} [Job: {job_id}] | Error: {e}")
        raise