import asyncio
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv(".env")

from google import genai
from google.genai import types
from app.core.config import get_settings

async def test_config(name, config_kwargs):
    client = genai.Client(vertexai=True, api_key=os.getenv("VERTEX_AI_API_KEY"))
    settings = get_settings()
    
    print(f"\n--- Testing: {name} ---")
    try:
        response = await client.aio.models.generate_content(
            model=settings.IMAGE_GEN_MODEL,
            contents='A cool car',
            config=types.GenerateContentConfig(**config_kwargs)
        )
        print("Response received: 200 OK")
        if response.parts:
            print("Parts:", len(response.parts))
            for p in response.parts:
                if p.inline_data:
                    print("  - Image part found")
                elif p.text:
                    print(f"  - Text part found: {p.text}")
                else:
                    print("  - Other part found")
        else:
            print("Response had NO PARTS!")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    settings = get_settings()
    ar = settings.VIDEO_ASPECT_RATIO
    sz = settings.IMAGE_GEN_SIZE
    print(f"Using AR: {ar}, Size: {sz}")

    await test_config("IMAGE modality only", {
        "response_modalities": ["IMAGE"],
    })
    
    await test_config("IMAGE modality + image_config", {
        "response_modalities": ["IMAGE"],
        "image_config": types.ImageConfig(aspect_ratio=ar, image_size=sz)
    })
    
    await test_config("TEXT, IMAGE modalities + image_config", {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": types.ImageConfig(aspect_ratio=ar, image_size=sz)
    })
    
    await test_config("No modalities + image_config", {
        "image_config": types.ImageConfig(aspect_ratio=ar, image_size=sz)
    })

asyncio.run(main())