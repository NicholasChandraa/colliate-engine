import asyncio
import os
from app.core.llm import get_genai_client
from google.genai import types

async def test_video_real():
    print("Testing Video Generation with Real App Configuration...")
    
    # This will load via config.py which injects os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    client = get_genai_client()
    
    print(f"Client project: {client._api_client.project}")
    print(f"Client location: {client._api_client.location}")
    print(f"Credentials loaded: {bool(client._api_client._credentials)}")
    
    import base64
    dummy_png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    primary_image = types.Image(image_bytes=dummy_png, mime_type="image/png")
    video_config = types.GenerateVideosConfig(
        duration_seconds=8,
        aspect_ratio="16:9",
        person_generation="ALLOW_ADULT",
    )
    
    try:
        print("Sending request to Veo...")
        operation = await client.aio.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt="A peaceful beach at sunset",
            image=primary_image,
            config=video_config,
        )
        print("Request successful! Operation name:", operation.name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_video_real())
