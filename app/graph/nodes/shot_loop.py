import asyncio
import json
import os
import re
import uuid
import wave
from google import genai
from google.genai import types
from typing import Any, cast
from app.core.config import get_settings
from app.core.exceptions import ImageGenerationError, VideoGenerationError, VideoRateLimitError, VideoSafetyFilterError, VideoQuotaExhaustedError
from app.core.logging import logger
from app.core.llm import get_genai_client, get_video_genai_client
from app.graph.state import GraphState

IMAGE_GEN_PROMPT_WITH_PRODUCT = """
Create a skincare product shot for a science-based educational video:

Scene composition: {image_prompt}

CRITICAL — Product fidelity rules:
- Use the EXACT product from the reference image. Do NOT change anything about the product:
  bottle shape, label design, label text, brand name, color, cap color, cap shape, or any detail.
- The reference product must appear identical — treat it as a hero product shot.
- Only change: background, lighting, props around the product, composition.
- Style: clean studio product photography, professional editorial skincare aesthetic.
- NO human faces, NO realistic human figures.
- No added text, no watermarks.
"""

IMAGE_GEN_PROMPT_NO_PRODUCT = """
Create a photorealistic macro/ingredient/formula photograph for a premium skincare brand video.

Scene description: {image_prompt}

Mandatory visual style — photorealistic commercial beauty photography:
- Style reference: Dior ingredient campaign, Lancôme serum commercial, Vogue Beauty editorial.
  Every shot must look like it was captured by a professional commercial photographer.
- NO scientific diagrams, NO stylized 3D illustrations, NO cartoon renders.
- NO human faces, NO realistic human figures, NO product packaging.
- No text, no watermarks, no labels.

Photography guidelines per scene type:
- Skin texture shots: DSLR macro lens, extreme close-up of skin surface only (no face),
  dramatic side lighting to emphasize texture detail, very shallow depth of field.
- Ingredient nature shots: food/nature editorial macro, fresh and vivid, backlight or soft
  studio light, like a high-end botanical illustration brought to life photographically.
- Extracted ingredient shots: clinical studio lighting, clean dark or white background,
  crystals/powder/liquid on a pristine surface, spotlight from above.
- Formula/serum shots: ultra-close macro of liquid texture, droplets, gel consistency,
  ASMR-quality detail, soft diffused light, transparent or glass surface preferred.
- Healthy skin shots: warm flattering light, smooth skin surface macro (no face),
  slight dewy glow, shallow depth of field, looks healthy and luminous.
"""


_REFERENCE_HINTS: dict[str, str] = {
    "product": (
        "\n\nADDITIONAL PRODUCT REFERENCE: A second reference image of the same product from a different angle "
        "is provided. Use both product images for accurate product representation — same label, branding, "
        "shape, and color must be consistent."
    ),
    "character": (
        "\n\nCHARACTER REFERENCE: A reference image of the model/character is provided. "
        "Use it for any human figure in this scene — maintain exact facial features, "
        "skin tone, hair color, and overall appearance for consistency across shots."
    ),
    "skin": (
        "\n\nSKIN REFERENCE: A reference skin texture/condition image is provided. "
        "Use it as visual reference for the skin appearance, texture, and tone in relevant shots. "
        "Match the skin quality and characteristics shown in the reference."
    ),
}


async def _generate_scene_image(
    client: genai.Client,
    shot: dict[str, object],
    product_bytes: bytes,
    reference_bytes: bytes = b"",
    reference_type: str = "",
) -> bytes:
    settings = get_settings()
    include_product = bool(shot.get("include_product", True))
    has_reference = len(reference_bytes) > 0
    image_prompt = cast(str, shot.get("image_prompt", ""))

    prompt_template = IMAGE_GEN_PROMPT_WITH_PRODUCT if include_product else IMAGE_GEN_PROMPT_NO_PRODUCT
    prompt = prompt_template.format(image_prompt=image_prompt)

    if has_reference and reference_type in _REFERENCE_HINTS:
        prompt += _REFERENCE_HINTS[reference_type]

    contents: list[types.Part] = []
    if has_reference:
        contents.append(types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"))
    if include_product:
        contents.append(types.Part.from_bytes(data=product_bytes, mime_type="image/jpeg"))
    contents.append(types.Part.from_text(text=prompt))

    try:
        logger.info(f"📷 PROMPT IMAGE GENERATION [include_product={include_product}]: {prompt}")
        response = await client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=settings.IMAGE_GEN_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],  # type: ignore[call-arg]
                image_config=types.ImageConfig(  # type: ignore[call-arg]
                    aspect_ratio=cast(str, settings.VIDEO_ASPECT_RATIO),
                    image_size=cast(str, settings.IMAGE_GEN_SIZE),
                ),
            ),
        )

        if not response.parts:
            raise ImageGenerationError(f"Shot {shot['id']}: image gen returned no parts")

        for part in response.parts:
            if part.inline_data and part.inline_data.data:
                logger.info(f"📸 Image Model Generated Scene [Shot {shot['id']}]")
                return cast(bytes, part.inline_data.data)

        raise ImageGenerationError(f"Shot {shot['id']}: image gen returned no inline_data in parts")

    except ImageGenerationError:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Image Model Failed [Shot {shot['id']}] -> Error: {error_str}")
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            if "exceeded your current quota" in error_str.lower():
                raise VideoQuotaExhaustedError(f"Shot {shot['id']}: Gemini API Quota Exhausted. Check billing.") from e
        raise ImageGenerationError(f"Shot {shot['id']}: image gen failed: {e}") from e


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

        # Vertex AI returns the video bytes directly in the response
        if getattr(video.video, "video_bytes", None):
            video_bytes = video.video.video_bytes
        else:
            raise VideoGenerationError(f"Shot {shot['id']}: downloaded video has no bytes (or only URI returned)")

        if not video_bytes:
            raise VideoGenerationError(f"Shot {shot['id']}: downloaded video has no bytes")

        video_path = os.path.join(output_dir, f"shot_{shot['id']:02d}.mp4")
        with open(video_path, "wb") as f:
            f.write(cast(bytes, video_bytes))

        logger.info(f"🎥 Video Model Completed [Shot {shot['id']}] -> Output: {video_path}")
        return video_path

    except VideoGenerationError:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Video Model Failed [Shot {shot['id']}] -> Error: {error_str}")
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            if "exceeded your current quota" in error_str.lower():
                raise VideoQuotaExhaustedError(f"Shot {shot['id']}: Gemini API Quota Exhausted. Check billing.") from e
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


async def _generate_tts_audio(
    client: genai.Client,
    voiceover_text: str,
    output_path: str,
) -> str:
    """
    Generate TTS voiceover from text using Gemini 2.5 Flash TTS.
    Returns the path to the saved .wav file.

    The Gemini TTS API returns raw PCM audio (24kHz, 16-bit, mono).
    We wrap it in a proper WAVE container using Python's built-in `wave` module.
    """
    settings = get_settings()

    try:
        logger.info(f"🎙️ TTS Generation Started -> '{voiceover_text[:60]}...'")
        response = await client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
            model=settings.TTS_MODEL,
            contents=voiceover_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],  # type: ignore[call-arg]
                speech_config=types.SpeechConfig(  # type: ignore[call-arg]
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=settings.TTS_VOICE_NAME,
                        )
                    )
                ),
            ),
        )

        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        parts = content.parts if content else None  # type: ignore[union-attr]
        if not parts:
            raise RuntimeError("TTS returned no audio parts")

        audio_part = parts[0]
        if not audio_part.inline_data or not audio_part.inline_data.data:
            raise RuntimeError("TTS returned empty inline_data")

        raw_pcm: bytes = cast(bytes, audio_part.inline_data.data)

        # Wrap raw PCM (24kHz, 16-bit, mono) in a .wav container
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)      # mono
            wav_file.setsampwidth(2)      # 16-bit = 2 bytes
            wav_file.setframerate(24000)  # 24kHz — Gemini TTS default
            wav_file.writeframes(raw_pcm)

        logger.info(f"✅ TTS Audio Saved -> {output_path} ({len(raw_pcm) / 1024:.1f} KB PCM)")
        return output_path

    except Exception as e:
        logger.error(f"❌ TTS Generation Failed -> {str(e)}")
        raise

async def _rewrite_blocked_shot(shot: dict[str, object]) -> dict[str, object]:
    """Ask LLM to rewrite a safety-filter-blocked shot prompt with safer language."""
    settings = get_settings()
    client = get_genai_client()

    prompt = REWRITE_PROMPT.format(
        type=shot.get("type", ""),
        subject_action=shot.get("subject_action", ""),
        emotion=shot.get("emotion", ""),
        video_prompt=shot.get("video_prompt", ""),
        negative_prompt=shot.get("negative_prompt", ""),
    )

    logger.info(f"🔄 Rewriting blocked shot {shot['id']} prompt via LLM...")
    response = await client.aio.models.generate_content(
        model=settings.LLM_RESEARCH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),  # pyright: ignore[reportCallIssue]
    )

    raw = response.text or ""
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    result = cast(dict[str, object], json.loads(raw))
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

    client = get_genai_client()
    video_client = get_video_genai_client()

    # Phase 1: Generate all scene images first
    logger.info("shot_loop_node.phase1 | generating all scene images")
    scene_images: list[bytes] = []
    for shot in state.storyboard:
        logger.info(f"shot_loop_node.scene_image | shot_id: {shot['id']} | type: {shot['type']}")
        scene_image = await _generate_scene_image(
            client=client,
            shot=shot,
            product_bytes=state.product_image_bytes,
        )
        scene_images.append(scene_image)

        img_path = os.path.join(output_dir, f"scene_{shot['id']:02d}.png")
        with open(img_path, "wb") as f:
            f.write(scene_image)
            
        # Add a brief delay to prevent triggering Vertex AI rate limits (burst limit)
        await asyncio.sleep(4)

    # Phase 2: Generate video clips with start+end frame pairs
    logger.info("shot_loop_node.phase2 | generating video clips")
    generated_video_paths: list[str] = []
    for i, (shot, scene_image) in enumerate(zip(state.storyboard, scene_images)):
        end_frame = scene_images[i + 1] if i + 1 < len(scene_images) else None
        logger.info(f"shot_loop_node.video_clip | shot_id: {shot['id']} | has_end_frame: {end_frame is not None}")

        try:
            video_path = await _generate_video_clip(
                client=video_client,
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
                    client=video_client,
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
