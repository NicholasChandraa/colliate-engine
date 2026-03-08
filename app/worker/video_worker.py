import os
from tenacity import RetryError
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
    reject_job,
    get_job,
    create_job_shots,
    update_job_shot_image,
    update_job_shot_video,
    update_job_shot_raw_video,
    update_job_shot_audio,
    get_selected_shots,
)


async def _check_cancellation(db, job_id: str) -> bool:
    job = await get_job(db, job_id)
    if job and job.status == JobStatus.CANCELLED:
        logger.warning(f"🛑 Job Cancelled mid-flight [Job: {job_id}] -> Halting pipeline.")
        return True
    return False


# ── Task 1: Research + Director + Image Generation ────────────────────────────

async def generate_images_task(
    ctx: dict,
    job_id: str,
    product_name: str,
    product_image_bytes: bytes,
    reference_image_bytes: bytes | None = None,
    reference_image_type: str | None = None,
) -> None:
    """
    ARQ Task 1 — Runs research, director, and generates 2 image options per shot.
    Ends with status AWAITING_SELECTION — waits for user to select images via API
    before Task 2 (generate_videos_task) is triggered.
    """
    logger.info(f"🚀 Task 1 Started [Job: {job_id}]")

    async with AsyncSessionFactory() as db:
        try:
            # ── Stage 1: Research ─────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.RESEARCHING,
                progress_message="Researching product...",
            )
            await db.commit()

            initial_state = GraphState(
                job_id=job_id,
                product_name=product_name,
                product_image_bytes=product_image_bytes or b"",
                reference_image_bytes=reference_image_bytes or b"",
                reference_image_type=reference_image_type or "",
            )

            if await _check_cancellation(db, job_id): return
            after_research = await research_node(initial_state)
            state = initial_state.model_copy(update=after_research)

            job = await get_job(db, job_id)
            if job:
                job.product_research = state.product_research
                await db.commit()

            # ── Verdict Check ─────────────────────────────────────────────
            if state.product_verdict == "REJECTED":
                logger.warning(f"🚫 Product Rejected [Job: {job_id}] -> {state.product_verdict_reason}")
                await reject_job(db, job_id, state.product_verdict_reason)
                await db.commit()
                return

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

            # ── Stage 3: Image Generation (2 images per shot) ─────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING_IMAGES,
                progress_message="Generating image options...",
            )
            await db.commit()

            if await _check_cancellation(db, job_id): return
            await _run_image_generation_with_progress(db, job_id, state, total_shots)

            # ── Awaiting Selection ────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.AWAITING_SELECTION,
                progress_message="Images ready. Select your preferred image for each shot.",
            )
            await db.commit()
            logger.info(f"⏸️ Task 1 Complete — Awaiting user selection [Job: {job_id}]")

        except VideoAdGeneratorError as e:
            logger.error(f"❌ Task 1 Failed [Job: {job_id}] -> {str(e)}")
            await fail_job(db, job_id, str(e))
            await db.commit()

        except RetryError as e:
            cause = str(e.last_attempt.exception()) if e.last_attempt else str(e)
            logger.error(f"❌ Task 1 Failed after retries [Job: {job_id}] -> {cause}")
            await fail_job(db, job_id, cause)
            await db.commit()

        except Exception as e:
            logger.error(f"💥 Task 1 Unexpected Crash [Job: {job_id}] -> {str(e)}")
            await fail_job(db, job_id, f"Unexpected error: {e}")
            await db.commit()


# ── Task 2: Video Generation + Assembly ───────────────────────────────────────

async def generate_videos_task(
    ctx: dict,
    job_id: str,
) -> None:
    """
    ARQ Task 2 — Triggered by POST /api/jobs/{job_id}/approve after user selects images.
    Generates video clips only for selected shots, then assembles the final video.
    """
    logger.info(f"🎬 Task 2 Started [Job: {job_id}]")

    async with AsyncSessionFactory() as db:
        try:
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING_VIDEOS,
                progress_message="Generating video clips for selected shots...",
            )
            await db.commit()

            if await _check_cancellation(db, job_id): return
            generated_video_paths = await _run_video_generation_with_progress(db, job_id)

            if not generated_video_paths:
                raise VideoAdGeneratorError("No video clips were generated — no shots had a valid selection.")

            # ── Assembly ──────────────────────────────────────────────────
            await update_job_status(
                db, job_id,
                status=JobStatus.ASSEMBLING,
                progress_message="Assembling final video...",
            )
            await db.commit()

            if await _check_cancellation(db, job_id): return

            # Reconstruct minimal state for assembly node
            settings = get_settings()
            assembly_state = GraphState(
                job_id=job_id,
                generated_video_paths=generated_video_paths,
            )
            after_assembly = assembly_node(assembly_state)
            final_video_path = str(after_assembly.get("final_video_path", ""))

            await complete_job(db, job_id, final_video_path)
            await db.commit()
            logger.info(f"✅ Task 2 Complete [Job: {job_id}] -> {final_video_path}")

        except VideoAdGeneratorError as e:
            logger.error(f"❌ Task 2 Failed [Job: {job_id}] -> {str(e)}")
            await fail_job(db, job_id, str(e))
            await db.commit()

        except RetryError as e:
            cause = str(e.last_attempt.exception()) if e.last_attempt else str(e)
            logger.error(f"❌ Task 2 Failed after retries [Job: {job_id}] -> {cause}")
            await fail_job(db, job_id, cause)
            await db.commit()

        except Exception as e:
            logger.error(f"💥 Task 2 Unexpected Crash [Job: {job_id}] -> {str(e)}")
            await fail_job(db, job_id, f"Unexpected error: {e}")
            await db.commit()


# ── Internal: Image Generation Phase ─────────────────────────────────────────

async def _run_image_generation_with_progress(
    db,
    job_id: str,
    state: GraphState,
    total_shots: int,
) -> None:
    """
    Generate 2 scene images per shot and save both paths to DB.
    """
    from google import genai
    from app.graph.nodes.shot_loop import _generate_scene_image

    settings = get_settings()
    client = genai.Client(
        vertexai=True,
        api_key=settings.VERTEX_AI_API_KEY,
    )
    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    for i, shot in enumerate(state.storyboard, start=1):
        if await _check_cancellation(db, job_id):
            return

        await update_job_status(
            db, job_id,
            status=JobStatus.GENERATING_IMAGES,
            completed_shots=i - 1,
            progress_message=f"Generating images for shot {i}/{total_shots}...",
        )
        await db.commit()

        # Image option 1
        image_1 = await _generate_scene_image(
            client=client,
            shot=shot,
            product_bytes=state.product_image_bytes,
            reference_bytes=state.reference_image_bytes,
            reference_type=state.reference_image_type,
        )
        img_path_1 = os.path.join(output_dir, f"scene_{shot['id']:02d}_a.png")
        with open(img_path_1, "wb") as f:
            f.write(image_1)
        await update_job_shot_image(db, job_id, shot_index=i, image_path=img_path_1, image_number=1)
        await db.commit()

        # Image option 2
        image_2 = await _generate_scene_image(
            client=client,
            shot=shot,
            product_bytes=state.product_image_bytes,
            reference_bytes=state.reference_image_bytes,
            reference_type=state.reference_image_type,
        )
        img_path_2 = os.path.join(output_dir, f"scene_{shot['id']:02d}_b.png")
        with open(img_path_2, "wb") as f:
            f.write(image_2)
        await update_job_shot_image(db, job_id, shot_index=i, image_path=img_path_2, image_number=2)
        await db.commit()

        # Mark this shot as completed for progress tracking
        await update_job_status(
            db, job_id,
            status=JobStatus.GENERATING_IMAGES,
            completed_shots=i,
        )
        await db.commit()

        logger.info(f"📸 Shot {i}/{total_shots} — both images saved [Job: {job_id}]")


# ── Internal: Video Generation Phase ─────────────────────────────────────────

async def _run_video_generation_with_progress(
    db,
    job_id: str,
) -> list[str]:
    """
    For each shot that the user selected an image for:
    - Load the selected image from filesystem
    - Generate a video clip (with next shot's image as end frame)
    - Save clip path to DB
    Returns list of generated video clip paths in shot order.
    """
    from app.core.exceptions import VideoSafetyFilterError
    from app.graph.nodes.shot_loop import _generate_video_clip, _generate_tts_audio, _rewrite_blocked_shot
    from app.graph.nodes.assembly import _merge_video_audio
    from app.core.llm import get_genai_client, get_video_genai_client

    settings = get_settings()
    client = get_genai_client()
    video_client = get_video_genai_client()
    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    selected_shots = await get_selected_shots(db, job_id)
    total = len(selected_shots)
    logger.info(f"🎬 Video generation starting — {total} selected shots [Job: {job_id}]")

    # Pre-load all selected images for end-frame pairing
    scene_images: list[bytes] = []
    for shot in selected_shots:
        img_path = shot.scene_image_path if shot.selected_image == 1 else shot.scene_image_path_2
        with open(img_path, "rb") as f:  # type: ignore[arg-type]
            scene_images.append(f.read())

    generated_video_paths: list[str] = []

    for i, (shot, scene_image) in enumerate(zip(selected_shots, scene_images)):
        if await _check_cancellation(db, job_id):
            return generated_video_paths

        end_frame = scene_images[i + 1] if i + 1 < len(scene_images) else None

        await update_job_status(
            db, job_id,
            status=JobStatus.GENERATING_VIDEOS,
            completed_shots=i,
            progress_message=f"Generating video clip {i + 1}/{total}: {shot.shot_type}...",
        )
        await db.commit()

        # Build shot dict for video generation (uses stored prompts from DB)
        shot_dict: dict[str, object] = {
            "id": shot.shot_index,
            "type": shot.shot_type,
            "subject_action": shot.subject_action or "",
            "emotion": shot.emotion or "",
            "video_prompt": shot.video_prompt or "",
            "negative_prompt": shot.negative_prompt or "",
        }

        if await _check_cancellation(db, job_id):
            return generated_video_paths

        try:
            video_path = await _generate_video_clip(
                client=video_client,
                shot=shot_dict,
                scene_image_bytes=scene_image,
                end_frame_bytes=end_frame,
                output_dir=output_dir,
            )
        except VideoSafetyFilterError:
            logger.warning(f"\U0001f6e1\ufe0f Shot {shot.shot_index} blocked by safety filter — rewriting prompt...")
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING_VIDEOS,
                progress_message=f"Shot {i + 1} blocked — rewriting prompt...",
            )
            await db.commit()
            rewritten = await _rewrite_blocked_shot(shot_dict)
            try:
                video_path = await _generate_video_clip(
                    client=video_client,
                    shot=rewritten,
                    scene_image_bytes=scene_image,
                    end_frame_bytes=end_frame,
                    output_dir=output_dir,
                )
            except (VideoSafetyFilterError, RetryError):
                logger.error(f"🚫 Shot {shot.shot_index} skipped — still blocked after rewrite.")
                await update_job_status(
                    db, job_id,
                    status=JobStatus.GENERATING_VIDEOS,
                    progress_message=f"Shot {i + 1} skipped (safety filter).",
                )
                await db.commit()
                continue
        except RetryError as e:
            # Rate limit hit — all retries exhausted. Skip this shot, keep going.
            cause = str(e.last_attempt.exception()) if e.last_attempt else str(e)
            logger.error(
                f"\u23f3 Shot {shot.shot_index} skipped — rate limit exhausted after retries. "
                f"Reason: {cause}"
            )
            await update_job_status(
                db, job_id,
                status=JobStatus.GENERATING_VIDEOS,
                progress_message=f"Shot {i + 1} skipped (rate limit). Continuing...",
            )
            await db.commit()
            continue

        # ── Save raw Veo clip path ──────────────────────────────────────
        await update_job_shot_raw_video(db, job_id, shot_index=shot.shot_index, raw_video_path=video_path)
        await db.commit()

        # ── TTS Voiceover + Merge ──────────────────────────────────────
        voiceover_text: str = shot.voiceover_text or ""
        if voiceover_text.strip():
            tts_path = os.path.join(output_dir, f"tts_shot_{shot.shot_index:02d}.wav")
            try:
                await update_job_status(
                    db, job_id,
                    status=JobStatus.GENERATING_VIDEOS,
                    progress_message=f"Generating voiceover for shot {i + 1}/{total}...",
                )
                await db.commit()
                await _generate_tts_audio(
                    client=client,
                    voiceover_text=voiceover_text,
                    output_path=tts_path,
                )
                await update_job_shot_audio(db, job_id, shot_index=shot.shot_index, audio_path=tts_path)
                await db.commit()
                # Merge video + TTS audio into a single clip
                merged_path = os.path.join(output_dir, f"merged_shot_{shot.shot_index:02d}.mp4")
                merged_path = _merge_video_audio(
                    video_path=video_path,
                    audio_path=tts_path,
                    output_path=merged_path,
                )
                video_path = merged_path
                logger.info(f"🎧 Shot {shot.shot_index} — video + TTS merged [Job: {job_id}]")
            except Exception as tts_err:
                logger.warning(
                    f"⚠️ TTS/merge failed for shot {shot.shot_index} — using video-only fallback. "
                    f"Error: {tts_err}"
                )
        else:
            logger.warning(f"⚠️ Shot {shot.shot_index} has no voiceover_text — skipping TTS.")

        generated_video_paths.append(video_path)
        await update_job_shot_video(db, job_id, shot_index=shot.shot_index, video_path=video_path)
        await db.commit()

    await update_job_status(db, job_id, status=JobStatus.GENERATING_VIDEOS, completed_shots=total)
    await db.commit()

    return generated_video_paths


# ── ARQ Worker Settings ────────────────────────────────────────────────────────

class WorkerSettings:
    functions = [generate_images_task, generate_videos_task]
    max_jobs = 3
    job_timeout = 7200
    max_tries = 1
    keep_result = 3600

    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL)
