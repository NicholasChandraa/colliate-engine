"""
Integration test — Research + Director + Imagen (Skincare Niche)

Runs the full pipeline up to image generation (Veo is NOT called):
  1. research_node  -> scientific brief + APPROVED/REJECTED verdict
  2. director_node  -> 8-shot skincare science storyboard
  3. _generate_scene_image (Imagen) for every shot in the storyboard

Usage (from engine/ directory):
    uv run python tests/test_imagen_skincare.py

Output images saved to: tests/output/<product_name>/
"""
import asyncio
import os
import sys
from google import genai
from app.core.config import get_settings
from app.graph.state import GraphState
from app.graph.nodes.research import research_node
from app.graph.nodes.director import director_node
from app.graph.nodes.shot_loop import _generate_scene_image

# ── Config ────────────────────────────────────────────────────────────────────

PRODUCT_NAME  = "skintific msh niacinamide brightening gel 3"  # ganti sesuai produk yang mau ditest

ASSETS_DIR    = os.path.join(os.path.dirname(__file__), "asset_test")
PRODUCT_IMAGE = os.path.join(ASSETS_DIR, "produk.png")
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")


# ── Test Runner ───────────────────────────────────────────────────────────────

async def run(product_bytes: bytes) -> None:
    settings = get_settings()

    # ── Stage 1: Research ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1: Research Node")
    print("="*60)

    state = GraphState(
        job_id="test-local",
        product_name=PRODUCT_NAME,
        product_image_bytes=product_bytes,
    )

    research_result = await research_node(state)
    state = state.model_copy(update=research_result)

    print(f"\nVerdict  : {state.product_verdict}")
    print(f"Reason   : {state.product_verdict_reason}")
    print(f"\nScientific Brief ({len(state.product_research)} chars):")
    print("-" * 40)
    print(state.product_research[:800] + "..." if len(state.product_research) > 800 else state.product_research)

    if state.product_verdict == "REJECTED":
        print(f"\n🚫 Product REJECTED — stopping test.")
        print(f"   Reason: {state.product_verdict_reason}")
        return

    print(f"\n✅ Product APPROVED — proceeding to Director.")

    # ── Stage 2: Director ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: Director Node")
    print("="*60)

    director_result = await director_node(state)
    state = state.model_copy(update=director_result)

    print(f"\nTotal shots: {len(state.storyboard)}")
    print("\nStoryboard summary:")
    for shot in state.storyboard:
        product_flag = "📦 WITH PRODUCT" if shot.get("include_product") else "🔬 pure generated"
        print(f"  Shot {shot['id']:02d} | {str(shot['type']):<20} | {product_flag}")
        print(f"         image_prompt: {str(shot.get('image_prompt', ''))[:80]}...")

    # ── Stage 3: Imagen ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: Imagen — generating scene images")
    print("="*60)

    safe_name = PRODUCT_NAME.replace(" ", "_").lower()
    shot_output_dir = os.path.join(OUTPUT_DIR, safe_name)
    os.makedirs(shot_output_dir, exist_ok=True)

    client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    passed = 0
    failed = 0

    for shot in state.storyboard:
        shot_id = shot["id"]
        shot_type = shot["type"]
        include_product = shot.get("include_product", False)
        label = f"Shot {shot_id:02d} [{shot_type}] include_product={include_product}"

        print(f"\n  [{label}]")
        try:
            image_bytes = await _generate_scene_image(
                client=client,
                shot=shot,
                product_bytes=product_bytes,
            )
            filename = f"shot_{shot_id:02d}_{shot_type}.png"
            out_path = os.path.join(shot_output_dir, filename)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            print(f"  ✅ Saved -> {out_path} ({len(image_bytes):,} bytes)")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED -> {e}")
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(state.storyboard)} shots")
    print(f"Output : {shot_output_dir}")
    if failed == 0:
        print("✅ All stages passed.")
    else:
        print("⚠️  Some image generations failed. Check output above.")


def main() -> None:
    if not os.path.exists(PRODUCT_IMAGE):
        print(f"❌ Product image not found: {PRODUCT_IMAGE}")
        print(f"   Place your product image at: {PRODUCT_IMAGE}")
        sys.exit(1)

    with open(PRODUCT_IMAGE, "rb") as f:
        product_bytes = f.read()

    print(f"Product  : {PRODUCT_NAME}")
    print(f"Image    : {PRODUCT_IMAGE} ({len(product_bytes):,} bytes)")

    asyncio.run(run(product_bytes))


if __name__ == "__main__":
    main()
