import asyncio
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")

async def test_video_studio():
    print("Testing AI Studio client (vertexai=False)...")
    
    # Use AI Studio (vertexai=False)
    client = genai.Client(
        vertexai=False,
        api_key=os.getenv("VERTEX_AI_API_KEY"),
    )
    
    try:
        models = await client.aio.models.list_models()
        print("Available models containing 'veo':")
        for m in models:
            if "veo" in m.name.lower():
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")

asyncio.run(test_video_studio())
