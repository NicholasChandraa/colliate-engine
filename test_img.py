import asyncio
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv(".env")

from google import genai
from google.genai import types

async def main():
    client = genai.Client(vertexai=True, api_key=os.getenv("VERTEX_AI_API_KEY"))
    
    prompt = """
    Create a photorealistic macro/ingredient/formula photograph for a premium skincare brand video.
    
    Scene description: Extreme macro photography of cheek area skin surface, young adult skin age 25-35, macro. Showing hyperpigmentation, dark spots, and uneven skin tone.
    Dramatic side lighting to emphasize the textured and pigmented surface. Soft blush pink and clean white color palette, clinical premium aesthetic. No realistic human faces, no full body.
    """
    
    print("Sending request...")
    try:
        response = await client.aio.models.generate_content(
            model='gemini-3.1-flash-image-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                    image_size="1920x1080"
                )
            )
        )
        print("Response received.")
        
        print("\nParts check:")
        if response.parts:
            print("Number of parts:", len(response.parts))
            for i, part in enumerate(response.parts):
                print(f"Part {i}:")
                if part.inline_data:
                    print("  Has inline_data:", True)
                    print("  Mime type:", part.inline_data.mime_type)
                    print("  Data length:", len(part.inline_data.data) if part.inline_data.data else 0)
                else:
                    print("  No inline_data")
                    print("  Text:", part.text)
        else:
            print("No parts in response")
            
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())