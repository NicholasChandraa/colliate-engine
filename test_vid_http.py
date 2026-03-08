import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(".env")

async def test_video_http():
    print("Testing Video Generation with API Key and HTTP Options...")
    
    # Pass api_key but use http_options to inject a project path header? 
    # Or set GOOGLE_CLOUD_PROJECT? Let's check what the API needs.
    client = genai.Client(
        vertexai=True,
        api_key=os.getenv("VERTEX_AI_API_KEY"),
        http_options={"headers": {"x-goog-user-project": "colliate"}}
    )
    
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

asyncio.run(test_video_http())
