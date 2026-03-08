import asyncio
import os
from dotenv import load_dotenv
from app.core.llm import get_video_genai_client
from app.graph.nodes.shot_loop import _generate_video_clip
from app.core.config import get_settings

# Load env file to ensure everything is picked up properly
load_dotenv(".env")

async def test_real_video_clip_generation():
    print("🎬 Testing REAL _generate_video_clip function from shot_loop.py...")
    
    settings = get_settings()
    video_client = get_video_genai_client()
    
    print(f"✅ Video Client Initialized: Project={video_client._api_client.project}, Location={video_client._api_client.location}")
    
    # 1. Create a dummy shot configuration simulating database values
    dummy_shot = {
        "id": 99, # Dummy ID
        "type": "hook",
        "video_prompt": "Slow panning camera motion. A peaceful beach at sunset with calm waves. Audio: Soft ocean breeze SFX, relaxing ambient music ONLY. No dialogue, no voiceover.",
        "negative_prompt": "realistic human faces, human body parts, text overlay, watermarks"
    }
    
    # 2. Create a dummy starting frame (1x1 black pixel PNG to keep it lightweight)
    import base64
    dummy_png_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    
    # 3. Prepare the output directory
    output_dir = os.path.join(settings.OUTPUT_DIR, "test_job_123")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Submitting Video Gen Request to Veo (This might take a few minutes)...")
    try:
        # 4. Call the exact function used in your LangGraph node
        video_path = await _generate_video_clip(
            client=video_client,
            shot=dummy_shot,
            scene_image_bytes=dummy_png_bytes,
            end_frame_bytes=None,
            output_dir=output_dir
        )
        
        print(f"\n🎉 SUCCESS! Video generated and saved successfully to:\n{video_path}")
        
    except Exception as e:
        print(f"\n❌ FAILED! Hit an error: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_video_clip_generation())
