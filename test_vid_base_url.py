import asyncio
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(".env")

async def test_video_base_url():
    print("Testing Video Generation with Custom Base URL...")
    
    project = os.getenv("GCP_PROJECT", "vertex-ai-project-489518")
    location = os.getenv("GCP_LOCATION", "us-central1")
    
    # Force the base URL to include the project and location path
    base_url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/"
    
    client = genai.Client(
        vertexai=True,
        api_key=os.getenv("VERTEX_AI_API_KEY"),
        http_options={"base_url": base_url}
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
        print(f"Sending request to Veo using base URL: {base_url} ...")
        operation = await client.aio.models.generate_videos(
            model="veo-3.1-generate-preview",
            prompt="A peaceful beach at sunset",
            image=primary_image,
            config=video_config,
        )
        print("Request successful! Operation name:", operation.name)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_video_base_url())
