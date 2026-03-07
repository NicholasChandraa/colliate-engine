"""
Integration test — Veo (frame-to-frame)

Loads pre-generated scene images from a test output folder and runs Veo
on each shot using start+end frame pairs (shot N → shot N+1).
Skips Imagen — images must already exist.

Usage:
    python tests/test_veo_integration.py

    # Or point to a different image folder:
    FRAMES_DIR=tests/output/my_product python tests/test_veo_integration.py

Output clips saved to: <FRAMES_DIR>/clips/
"""
import asyncio
import os
import sys
from google import genai
from app.core.config import get_settings
from app.graph.nodes.shot_loop import _generate_video_clip

# Default: use last skintific run. Override via env var FRAMES_DIR.
DEFAULT_FRAMES_DIR = os.path.join(
    os.path.dirname(__file__),
    "output",
    "skintific_msh_niacinamide_brightening_gel_3",
)
FRAMES_DIR = os.environ.get("FRAMES_DIR", DEFAULT_FRAMES_DIR)

# Filename → shot metadata (video_prompt + negative_prompt only; image_prompt not needed here)
SHOTS: list[dict[str, object]] = [
    {
        "id": 1,
        "type": "hook",
        "filename": "shot_01_hook.png",
        "subject_action": "Cheek area skin surface, enlarged pores and uneven texture visible",
        "emotion": "dramatic",
        "video_prompt": (
            "Slow dramatic push-in. Cheek area skin surface macro, young adult skin age 25-35. "
            "Enlarged pores and uneven texture highlighted by dramatic side lighting. "
            "Very shallow depth of field, soft bokeh background. "
            "Voiceover: \"Udah pakai skincare tiap hari tapi kulit masih kusam?\" "
            "Dramatic side lighting. All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, full face, human body parts, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 2,
        "type": "agitation",
        "filename": "shot_02_agitation.png",
        "subject_action": "Cheek area skin surface extreme macro, clogged pores close-up",
        "emotion": "clinical",
        "video_prompt": (
            "Extreme macro zoom-in. Cheek area skin surface, young adult skin age 25-35. "
            "Clogged pores and dead skin cells in extreme close-up. "
            "Dramatic side lighting, very narrow depth of field. "
            "Soft molecular zoom SFX. All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, full face, human body parts, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 3,
        "type": "ingredient_origin",
        "filename": "shot_03_ingredient_origin.png",
        "subject_action": "Niacinamide source — glowing rice grains in natural light",
        "emotion": "scientific",
        "video_prompt": (
            "Slow pull-back from extreme macro. Glowing rice grains in warm natural backlight, "
            "soft golden bokeh. Food editorial macro photography style. "
            "Calm clinical background music, subtle nature ambiance. "
            "All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, human body parts, product packaging, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 4,
        "type": "ingredient_extracted",
        "filename": "shot_04_ingredient_extracted.png",
        "subject_action": "Niacinamide crystals on dark surface, soft spotlight from above",
        "emotion": "premium",
        "video_prompt": (
            "Slow rotation around niacinamide crystals on dark surface. "
            "Soft spotlight from above, crystals catching the light and scattering reflections. "
            "Clinical premium aesthetic. "
            "Calm clinical background music. All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, human body parts, product packaging, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 5,
        "type": "formula_sensorial",
        "filename": "shot_05_formula_sensorial.png",
        "subject_action": "White gel formula on glass surface, elastic texture stretching",
        "emotion": "sensorial",
        "video_prompt": (
            "Slow macro pull-back. White lightweight gel formula on transparent glass, "
            "elastic texture slowly stretching as it spreads. "
            "Soft diffused studio lighting, clean white background. "
            "ASMR-quality detail. Subtle texture sound. "
            "All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, human body parts, product packaging, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 6,
        "type": "product_reveal",
        "filename": "shot_06_product_reveal.png",
        "subject_action": "Product on white marble surface, soft studio lighting",
        "emotion": "premium",
        "video_prompt": (
            "Camera slowly orbits the product. Clean white marble surface, "
            "soft top-down studio lighting, minimal composition. "
            "Premium editorial skincare aesthetic. "
            "Calm background music, subtle reveal tone. "
            "All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, human body parts, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 7,
        "type": "result",
        "filename": "shot_07_result.png",
        "subject_action": "Cheek area skin surface healthy and glowing, smooth texture",
        "emotion": "hopeful",
        "video_prompt": (
            "Slow gentle pull-back. Cheek area skin surface, young adult skin age 25-35. "
            "Smooth texture, small pores, even tone, subtle moisture glow. "
            "Soft warm flattering light. "
            "Voiceover: \"Kulit cerah, halus, dan sehat.\" "
            "All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, full face, human body parts, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
    {
        "id": 8,
        "type": "cta",
        "filename": "shot_08_cta.png",
        "subject_action": "Product on soft pink background with botanical props",
        "emotion": "empowering",
        "video_prompt": (
            "Slow gentle zoom-out. Product on soft pink background with small botanical props, "
            "warm diffused light, lifestyle feel. "
            "Voiceover: \"Coba sekarang dan rasakan perbedaannya.\" "
            "Warm uplifting background music. "
            "All audio and dialogue in Indonesian."
        ),
        "negative_prompt": (
            "realistic human faces, human body parts, text overlay, "
            "watermarks, blurry motion, distorted shapes, camera shake."
        ),
    },
]


async def run_test(frames_dir: str) -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    clips_dir = os.path.join(frames_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Load all frames upfront
    frames: list[bytes] = []
    for shot in SHOTS:
        path = os.path.join(frames_dir, str(shot["filename"]))
        if not os.path.exists(path):
            print(f"❌ Frame not found: {path}")
            sys.exit(1)
        with open(path, "rb") as f:
            frames.append(f.read())
    print(f"✅ Loaded {len(frames)} frames from {frames_dir}\n")

    # Generate video clip per shot
    failed = 0
    for i, shot in enumerate(SHOTS):
        end_frame = frames[i + 1] if i + 1 < len(frames) else None
        print(f"[{i+1}/{len(SHOTS)}] Shot {shot['id']} ({shot['type']}) — end_frame: {'yes' if end_frame else 'none'}")
        try:
            video_path = await _generate_video_clip(
                client=client,
                shot=shot,
                scene_image_bytes=frames[i],
                end_frame_bytes=end_frame,
                output_dir=clips_dir,
            )
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"      ✅ Saved -> {video_path} ({size_mb:.1f} MB)\n")
        except Exception as e:
            print(f"      ❌ FAILED: {e}\n")
            failed += 1

    print(f"{'✅ All clips generated.' if failed == 0 else f'⚠️ {failed}/{len(SHOTS)} shots failed.'}")
    print(f"Output: {clips_dir}")


def main() -> None:
    if not os.path.isdir(FRAMES_DIR):
        print(f"Frames directory not found: {FRAMES_DIR}")
        print("Set FRAMES_DIR env var to point to a folder with shot_0X_*.png files.")
        sys.exit(1)

    print(f"Frames dir : {FRAMES_DIR}")
    asyncio.run(run_test(FRAMES_DIR))


if __name__ == "__main__":
    main()
