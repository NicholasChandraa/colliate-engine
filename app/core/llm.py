from typing import Any
from google import genai
from app.core.config import get_settings

def get_genai_client() -> genai.Client:
    """
    Client for Text, Image, and TTS generation.
    Uses Google AI Studio API Key (vertexai=True but no project).
    This allows accessing models like gemini-3-flash-preview which might not be 
    enabled or available on the GCP Service Account yet.
    """
    settings = get_settings()
    return genai.Client(
        vertexai=True,
        api_key=settings.VERTEX_AI_API_KEY
    )

def get_image_genai_client() -> genai.Client:
    """
    Client strictly for Image generation (gemini-3.1-flash-image-preview).
    This model only supports the GLOBAL endpoint — regional endpoints return 404.
    Uses project + ADC so quota is tracked in the user's GCP project.
    """
    settings = get_settings()
    if settings.GCP_PROJECT:
        return genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT,
            location="global",
        )
    return genai.Client(
        vertexai=True,
        api_key=settings.VERTEX_AI_API_KEY,
    )


def get_video_genai_client() -> genai.Client:
    """
    Client strictly for Veo Video Generation.
    Uses Google Cloud Service Account (project & location).
    This is required because the predictLongRunning API for Veo 
    does not support API Keys.
    """
    settings = get_settings()
    
    if settings.GCP_PROJECT:
        return genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT,
            location=settings.GCP_LOCATION
        )
        
    # Fallback just in case (though it will fail for video)
    return genai.Client(
        vertexai=True,
        api_key=settings.VERTEX_AI_API_KEY
    )

