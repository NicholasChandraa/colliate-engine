import os
from tenacity import RetryError
from arq import ArqRedis
from arq.connections import RedisSettings
from app.core.config import get_settings
from app.core.database import AsyncSessionFactory
from app.core.logging import logger
from app.core.exceptions import VideoAdGeneratorError
from app.graph.state import GraphState
from app.graph.nodes.research import research_node
from app.graph.nodes.director import director_node
from app.graph.nodes.assembly import assembly_node
from app.models.job import JobStatus
from app.services.job_service import (
    update_job_status,
    complete_job,
    fail_job,
    get_job,
    create_job_shots,
    update_job_shot_image,
    update_job_shot_video,
)

async def _check_cancellation(db, job_id: str) -> bool:
    job = await get_job(db, job_id)
    if job and job.status == JobStatus.CANCELLED:
        logger.warning(f"🛑 Job Cancelled mid-flight [Job: {job_id}] -> Halting pipeline.")
        return True
    return False

async def generate_video_task(
    ctx: dict,
    job_id: str,
    product_name: str,
    target_audience: str,
    character_image_bytes: bytes,
    product_image_bytes: bytes,
) -> None:
    """
    ARQ background task — runs the full LangGraph pipeline.
    Updates job status in DB at each stage.
    """
    logger.info(f"🚀 Pipeline Initiated [Job: {job_id}]")

    async with AsyncSessionFactory() as db:
        try:
            # ── Stage 1: Research ─────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.RESEARCHING,
                progress_message="Researching product info...",
            )
            await db.commit()

            initial_state = GraphState(
                job_id=job_id,
                product_name=product_name,
                target_audience=target_audience,
                character_image_bytes=character_image_bytes,
                product_image_bytes=product_image_bytes,
            )

            if await _check_cancellation(db, job_id): return
            after_research = await research_node(initial_state)
            state = initial_state.model_copy(update=after_research)
            
            job = await get_job(db, job_id)
            if job:
                job.product_research = state.product_research
                await db.commit()

            # ── Stage 2: Director ─────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.DIRECTING,
                progress_message="Creating storyboard...",
            )
            await db.commit()

            if await _check_cancellation(db, job_id): return
            after_director = await director_node(state)
            state = state.model_copy(update=after_director)

            total_shots = len(state.storyboard)
            await update_job_status(
                db, job_id,
                status=JobStatus.DIRECTING,
                total_shots=total_shots,
                progress_message=f"Storyboard ready — {total_shots} shots planned.",
            )
            await create_job_shots(db, job_id, state.storyboard)
            await db.commit()

            # ── Stage 3: Shot Generation ──────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING,
                progress_message="Generating shots...",
            )
            await db.commit()

            state = await _run_shot_loop_with_progress(db, job_id, state, total_shots)

            # ── Stage 4: Assembly ─────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.ASSEMBLING,
                progress_message="Assembling final video...",
            )
            await db.commit()

            if await _check_cancellation(db, job_id): return
            after_assembly = assembly_node(state)
            state = state.model_copy(update=after_assembly)

            # ── Done ──────────────────────────────────────────────────────
            await complete_job(db, job_id, state.final_video_path)
            await db.commit()

            logger.info(f"✅ Pipeline Completed [Job: {job_id}] -> Output: {state.final_video_path}")

        except VideoAdGeneratorError as e:
            logger.error(f"❌ Pipeline Failed [Job: {job_id}] -> Error: {str(e)}")
            await fail_job(db, job_id, str(e))
            await db.commit()

        except RetryError as e:
            cause = str(e.last_attempt.exception()) if e.last_attempt else str(e)
            logger.error(f"❌ Pipeline Failed after retries [Job: {job_id}] -> Error: {cause}")
            await fail_job(db, job_id, cause)
            await db.commit()

        except Exception as e:
            logger.error(f"💥 Unexpected Pipeline Crash [Job: {job_id}] -> Error: {str(e)}")
            await fail_job(db, job_id, f"Unexpected error: {e}")
            await db.commit()


async def _run_shot_loop_with_progress(
    db,
    job_id: str,
    state: GraphState,
    total_shots: int,
) -> GraphState:
    """
    Run shot generation shot-by-shot so we can update DB progress per shot.
    Mirrors the logic in shot_loop_node but adds DB progress hooks.
    """
    from google import genai
    from app.core.config import get_settings
    from app.core.exceptions import VideoSafetyFilterError
    from app.graph.nodes.shot_loop import _generate_scene_image, _generate_video_clip, _rewrite_blocked_shot

    settings = get_settings()
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Generate all scene images
    scene_images: list[bytes] = []
    for i, shot in enumerate(state.storyboard, start=1):
        if await _check_cancellation(db, job_id):
            return state

        await update_job_status(
            db, job_id,
            status=JobStatus.GENERATING,
            progress_message=f"Generating scene image {i}/{total_shots}...",
        )
        await db.commit()

        scene_image = await _generate_scene_image(
            client=client,
            shot=shot,
            character_bytes=state.character_image_bytes,
            product_bytes=state.product_image_bytes,
        )
        scene_images.append(scene_image)

        img_path = os.path.join(output_dir, f"scene_{shot['id']:02d}.png")
        with open(img_path, "wb") as f:
            f.write(scene_image)
            
        await update_job_shot_image(db, job_id, shot_index=i, image_path=img_path)
        await db.commit()

    # Phase 2: Generate video clips with start+end frame pairs
    generated_video_paths: list[str] = []
    for i, (shot, scene_image) in enumerate(zip(state.storyboard, scene_images)):
        if await _check_cancellation(db, job_id):
            return state.model_copy(update={"generated_video_paths": generated_video_paths})

        end_frame = scene_images[i + 1] if i + 1 < len(scene_images) else None
        await update_job_status(
            db, job_id,
            status=JobStatus.GENERATING,
            completed_shots=i,
            progress_message=f"Generating video clip {i + 1}/{total_shots}: {shot['type']}...",
        )
        await db.commit()

        try:
            video_path = await _generate_video_clip(
                client=client,
                shot=shot,
                scene_image_bytes=scene_image,
                end_frame_bytes=end_frame,
                output_dir=output_dir,
            )
        except VideoSafetyFilterError:
            logger.warning(f"🛡️ Shot {shot['type']} blocked by safety filter — rewriting prompt and retrying...")
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING,
                progress_message=f"Shot {i + 1} blocked — rewriting prompt...",
            )
            await db.commit()
            rewritten_shot = await _rewrite_blocked_shot(shot)
            try:
                video_path = await _generate_video_clip(
                    client=client,
                    shot=rewritten_shot,
                    scene_image_bytes=scene_image,
                    end_frame_bytes=end_frame,
                    output_dir=output_dir,
                )
            except VideoSafetyFilterError:
                logger.error(f"🚫 Shot {i + 1} skipped — still blocked after rewrite. Continuing...")
                await update_job_status(
                    db, job_id,
                    status=JobStatus.GENERATING,
                    progress_message=f"Shot {i + 1} skipped (safety filter). Continuing...",
                )
                await db.commit()
                continue
        generated_video_paths.append(video_path)
        
        await update_job_shot_video(db, job_id, shot_index=i + 1, video_path=video_path)
        await db.commit()

    await update_job_status(db, job_id, status=JobStatus.GENERATING, completed_shots=total_shots)
    await db.commit()

    return state.model_copy(update={"generated_video_paths": generated_video_paths})


# ── ARQ Worker Settings ────────────────────────────────────────────────────────

class WorkerSettings:
    """ARQ worker configuration — pointed at by `arq src.workers.video_worker.WorkerSettings`."""

    functions = [generate_video_task]
    max_jobs = 3                # max concurrent video generation jobs
    job_timeout = 7200          # 2 hours max per job (Veo can be slow)
    max_tries = 1               # MUST NOT retry automatically on failure (too expensive!)
    keep_result = 3600          # keep result in Redis for 1 hour

    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL)
