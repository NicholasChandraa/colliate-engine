import asyncio
import os
import uuid
from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from typing import Any, cast
from app.core.config import get_settings
from app.core.exceptions import ImageGenerationError, VideoGenerationError, VideoRateLimitError, VideoSafetyFilterError
from app.core.logging import logger
from app.graph.state import GraphState

def _build_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.GOOGLE_API_KEY)

IMAGE_GEN_PROMPT_WITH_PRODUCT = """
Create a single photorealistic scene image for a product advertisement:

Scene description: {image_prompt}

Rules:
- The character from the FIRST reference image MUST appear in this scene exactly as they look
- The product from the SECOND reference image MUST be clearly visible in the scene
- Professional photography quality
- Cinematic lighting consistent with the scene description
- No text, no watermarks
"""

IMAGE_GEN_PROMPT_NO_PRODUCT = """
Create a single photorealistic scene image for a product advertisement:

Scene description: {image_prompt}

Rules:
- The character from the reference image MUST appear in this scene exactly as they look
- Focus entirely on the character, their emotion, and the environment — no product needed
- Professional photography quality
- Cinematic lighting consistent with the scene description
- No text, no watermarks
"""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(
        f"⚠️ Image Gen Failed (Attempt {retry_state.attempt_number}). Retrying..."
    )
)
async def _generate_scene_image(
    client: genai.Client,
    shot: dict[str, object],
    character_bytes: bytes,
    product_bytes: bytes,
) -> bytes:
    settings = get_settings()
    include_product = bool(shot.get("include_product", True))
    image_prompt = cast(str, shot.get("image_prompt", ""))

    prompt_template = IMAGE_GEN_PROMPT_WITH_PRODUCT if include_product else IMAGE_GEN_PROMPT_NO_PRODUCT
    prompt = prompt_template.format(image_prompt=image_prompt)

    contents: list[types.Part] = [types.Part.from_bytes(data=character_bytes, mime_type="image/jpeg")]
    if include_product:
        contents.append(types.Part.from_bytes(data=product_bytes, mime_type="image/jpeg"))
    contents.append(types.Part.from_text(text=prompt))

    try:
        logger.info(f"📷 PROMPT IMAGE GENERATION [include_product={include_product}]: {prompt}")
        response = await client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=settings.IMAGE_GEN_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=cast(str, settings.VIDEO_ASPECT_RATIO),
                    image_size=cast(str, settings.IMAGE_GEN_SIZE),
                ),
            ),
        )

        if not response.parts or not response.parts[0].inline_data:
            raise ImageGenerationError(f"Shot {shot['id']}: image gen returned no data")

        logger.info(f"📸 Image Model Generated Scene [Shot {shot['id']}]")
        return cast(bytes, response.parts[0].inline_data.data)

    except ImageGenerationError:
        raise
    except Exception as e:
        logger.error(f"❌ Image Model Failed [Shot {shot['id']}] -> Error: {str(e)}")
        raise ImageGenerationError(f"Shot {shot['id']}: image gen failed: {e}") from e


def _video_retry_wait(retry_state: object) -> float:
    """Wait 65s for rate limit errors, short exponential backoff for everything else."""
    exc = retry_state.outcome.exception()  # type: ignore[union-attr]
    if isinstance(exc, VideoRateLimitError):
        logger.warning(f"⏳ Rate limit hit — waiting 65s before retry...")
        return 65.0
    attempt = retry_state.attempt_number  # type: ignore[union-attr]
    return min(4.0 * (2 ** (attempt - 1)), 10.0)

@retry(
    stop=stop_after_attempt(2),
    wait=_video_retry_wait,
    retry=retry_if_exception_type(VideoGenerationError),  # does NOT retry VideoSafetyFilterError
    before_sleep=lambda retry_state: logger.warning(
        f"⚠️ Video Gen Failed (Attempt {retry_state.attempt_number}). Retrying..."
    )
)
async def _generate_video_clip(
    client: genai.Client,
    shot: dict[str, object],
    scene_image_bytes: bytes,
    end_frame_bytes: bytes | None,
    output_dir: str,
) -> str:
    settings = get_settings()
    primary_image = types.Image(image_bytes=scene_image_bytes, mime_type="image/png")
    end_frame_image = types.Image(image_bytes=end_frame_bytes, mime_type="image/png") if end_frame_bytes else None
    negative_prompt = cast(str, shot.get("negative_prompt", "")) or None
    # 1080p and 4K only support 8s duration per Veo API constraint
    video_config = types.GenerateVideosConfig(
        duration_seconds=settings.VIDEO_SHOT_DURATION_SECONDS,
        aspect_ratio=settings.VIDEO_ASPECT_RATIO,
        resolution=settings.VIDEO_RESOLUTION,
        number_of_videos=1,
        last_frame=end_frame_image,
        negative_prompt=negative_prompt,
    )
    prompt = cast(str, shot.get("video_prompt", ""))

    try:
        logger.info(f"📹 PROMPT VIDEO GENERATION: {prompt}")
        operation = await client.aio.models.generate_videos(
            model=settings.VIDEO_GEN_MODEL,
            prompt=prompt,
            image=primary_image,
            config=video_config,
        )

        while not operation.done:
            logger.info(f"⏳ Video Model Polling [Shot {shot['id']}]...")
            await asyncio.sleep(10)
            operation = await client.aio.operations.get(operation)

        if operation.error:
            error_message = cast(dict[str, object], operation.error).get("message", "Unknown error")
            logger.error(f"❌ Veo operation error shot_id={shot['id']} -> {error_message}")
            raise VideoGenerationError(f"Shot {shot['id']}: {error_message}")

        if not operation.response or not operation.response.generated_videos:
            raise VideoSafetyFilterError(
                f"Shot {shot['id']}: Veo returned no video — blocked by safety filter. Not retrying."
            )

        video = operation.response.generated_videos[0]

        if not video.video:
            raise VideoGenerationError(f"Shot {shot['id']}: video gen returned no file reference")

        # Veo stores video on Google servers — download() returns the raw bytes.
        # cast(Any) needed because SDK type stubs declare str|File but Video works at runtime.
        video_bytes = await client.aio.files.download(file=cast(Any, video.video))

        if not video_bytes:
            raise VideoGenerationError(f"Shot {shot['id']}: downloaded video has no bytes")

        video_path = os.path.join(output_dir, f"shot_{shot['id']:02d}.mp4")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        logger.info(f"🎥 Video Model Completed [Shot {shot['id']}] -> Output: {video_path}")
        return video_path

    except VideoGenerationError:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Video Model Failed [Shot {shot['id']}] -> Error: {error_str}")
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise VideoRateLimitError(f"Shot {shot['id']}: rate limit hit, will retry after delay") from e
        raise VideoGenerationError(f"Shot {shot['id']}: video gen failed: {e}") from e


REWRITE_PROMPT = """The following video_prompt was blocked by Veo 3.1 safety filters.
Rewrite it with safer language while preserving the exact same scene intent, camera motion, and audio cues.

Shot context:
- type: {type}
- subject_action: {subject_action}
- emotion: {emotion}

Original video_prompt:
{video_prompt}

Original negative_prompt:
{negative_prompt}

Rules:
- Preserve the scene intent and character action completely
- Avoid explicit descriptions of body parts, skin conditions, or appearance criticism
- Keep all camera motion terminology and audio cues intact
- Return ONLY valid JSON, no markdown: {{"video_prompt": "...", "negative_prompt": "..."}}"""


async def _rewrite_blocked_shot(shot: dict[str, object]) -> dict[str, object]:
    """Ask LLM to rewrite a safety-filter-blocked shot prompt with safer language."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import JsonOutputParser
    from typing import cast as _cast

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_RESEARCH_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3,
    )
    parser = JsonOutputParser()

    prompt = REWRITE_PROMPT.format(
        type=shot.get("type", ""),
        subject_action=shot.get("subject_action", ""),
        emotion=shot.get("emotion", ""),
        video_prompt=shot.get("video_prompt", ""),
        negative_prompt=shot.get("negative_prompt", ""),
    )

    logger.info(f"🔄 Rewriting blocked shot {shot['id']} prompt via LLM...")
    response = await llm.ainvoke(prompt)

    content = response.content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        json_str = _cast(str, content[0].get("text", ""))
    else:
        json_str = str(content)

    result = _cast(dict[str, object], parser.parse(json_str))
    logger.info(f"✅ Shot {shot['id']} prompt rewritten successfully.")

    return {
        **shot,
        "video_prompt": result.get("video_prompt", shot.get("video_prompt")),
        "negative_prompt": result.get("negative_prompt", shot.get("negative_prompt")),
    }


async def shot_loop_node(state: GraphState) -> dict[str, object]:
    logger.info(f"🎬 Graph Node: Shot Loop Engine Started -> Total Shots: {len(state.storyboard)}")

    settings = get_settings()
    run_id = state.job_id or str(uuid.uuid4())
    output_dir = os.path.join(settings.OUTPUT_DIR, run_id)
    os.makedirs(output_dir, exist_ok=True)

    client = _build_client()

    # Phase 1: Generate all scene images first
    logger.info("shot_loop_node.phase1 | generating all scene images")
    scene_images: list[bytes] = []
    for shot in state.storyboard:
        logger.info(f"shot_loop_node.scene_image | shot_id: {shot['id']} | type: {shot['type']}")
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

    # Phase 2: Generate video clips with start+end frame pairs
    logger.info("shot_loop_node.phase2 | generating video clips")
    generated_video_paths: list[str] = []
    for i, (shot, scene_image) in enumerate(zip(state.storyboard, scene_images)):
        end_frame = scene_images[i + 1] if i + 1 < len(scene_images) else None
        logger.info(f"shot_loop_node.video_clip | shot_id: {shot['id']} | has_end_frame: {end_frame is not None}")

        try:
            video_path = await _generate_video_clip(
                client=client,
                shot=shot,
                scene_image_bytes=scene_image,
                end_frame_bytes=end_frame,
                output_dir=output_dir,
            )
        except VideoSafetyFilterError:
            logger.warning(f"🛡️ Shot {shot['id']} blocked by safety filter — rewriting prompt and retrying...")
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
                logger.error(f"🚫 Shot {shot['id']} skipped — still blocked after rewrite. Continuing...")
                continue
        generated_video_paths.append(video_path)

    logger.info(f"🏁 Graph Node: Shot Loop Engine Finished -> Total Clips Mastered: {len(generated_video_paths)}")
    return {"generated_video_paths": generated_video_paths}
