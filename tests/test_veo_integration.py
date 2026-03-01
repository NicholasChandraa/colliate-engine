"""
Integration test — Veo 3.1 & Imagen (Nano Banana)

Verifies that _generate_scene_image and _generate_video_clip work end-to-end
against the real Google API.

Usage:
    python tests/test_veo_integration.py

Output files saved to: tests/output/
"""
import asyncio
import os
import sys
from google import genai
from app.core.config import get_settings
from app.graph.nodes.shot_loop import _generate_scene_image, _generate_video_clip

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "asset_test")
CHARACTER_IMAGE = os.path.join(ASSETS_DIR, "karakter.png")
PRODUCT_IMAGE   = os.path.join(ASSETS_DIR, "produk.png")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "output")

SAMPLE_SHOT: dict[str, object] = {
    "id": 1,
    "type": "hook",
    "camera_angle": "medium",
    "camera_movement": "slow push in",
    "subject_action": "holds the product and smiles toward the camera",
    "lighting": "soft natural window light from the right",
    "emotion": "confident",
    "image_prompt": (
        "Person standing in a bright modern living room. "
        "They hold the product clearly in front of them. "
        "Soft natural light from the right. Photorealistic, static single frame."
    ),
    "video_prompt": (
        "Camera slowly pushes in toward the person. "
        "They smile and gently raise the product. "
        "Soft bokeh background. Photorealistic motion. "
        "All audio and dialogue in Indonesian."
    ),
}


async def run_test(character_bytes: bytes, product_bytes: bytes) -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Generate scene image (Imagen / Nano Banana) ───────────────────
    print("\n[1/2] Generating scene image via Imagen...")
    try:
        scene_image = await _generate_scene_image(
            client=client,
            shot=SAMPLE_SHOT,
            character_bytes=character_bytes,
            product_bytes=product_bytes,
        )
        img_path = os.path.join(OUTPUT_DIR, "test_scene.png")
        with open(img_path, "wb") as f:
            f.write(scene_image)
        print(f"      ✅ Scene image saved -> {img_path} ({len(scene_image):,} bytes)")
    except Exception as e:
        print(f"      ❌ Scene image FAILED: {e}")
        return

    # ── Step 2: Generate video clip (Veo 3.1) ─────────────────────────────────
    print("\n[2/2] Generating video clip via Veo 3.1...")
    try:
        video_path = await _generate_video_clip(
            client=client,
            shot=SAMPLE_SHOT,
            scene_image_bytes=scene_image,
            end_frame_bytes=None,
            output_dir=OUTPUT_DIR,
        )
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"      ✅ Video clip saved -> {video_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"      ❌ Video clip FAILED: {e}")
        return

    print("\n✅ All steps passed. Integration test complete.")


def main() -> None:
    for path in (CHARACTER_IMAGE, PRODUCT_IMAGE):
        if not os.path.exists(path):
            print(f"Asset not found: {path}")
            sys.exit(1)

    with open(CHARACTER_IMAGE, "rb") as f:
        character_bytes = f.read()
    with open(PRODUCT_IMAGE, "rb") as f:
        product_bytes = f.read()

    print(f"Character image : {CHARACTER_IMAGE} ({len(character_bytes):,} bytes)")
    print(f"Product image   : {PRODUCT_IMAGE} ({len(product_bytes):,} bytes)")

    asyncio.run(run_test(character_bytes, product_bytes))


if __name__ == "__main__":
    main()
