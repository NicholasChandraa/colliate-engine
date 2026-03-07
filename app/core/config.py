from typing import ClassVar
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Google API
    GOOGLE_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/video_ad_db"

    # Redis / ARQ
    REDIS_URL: str = "redis://localhost:6379"

    # Models
    LLM_RESEARCH_MODEL: str = "gemini-3-flash-preview"
    LLM_DIRECTOR_MODEL: str = "gemini-3.1-pro-preview"
    LLM_DIRECTOR_MODEL_FALLBACK: str = "gemini-3-flash-preview"
    IMAGE_GEN_MODEL: str = "gemini-3.1-flash-image-preview"
    VIDEO_GEN_MODEL: str = "veo-3.1-generate-preview"
    THINKING_LEVEL: str = "low"
    
    # Video
    VIDEO_ASPECT_RATIO: str = "9:16"
    VIDEO_RESOLUTION: str = "1080p"
    VIDEO_SHOT_DURATION_SECONDS: int = 8
    VIDEO_TARGET_DURATION_SECONDS: int = 60
    VIDEO_LANGUAGE: str = "Indonesian"  # Language for Veo audio generation: "Indonesian" | "English"

    # Image generation
    IMAGE_GEN_SIZE: str = "1K"  # accepted: 512px | 1K | 2K | 4K (uppercase K required)

    # TTS
    TTS_MODEL: str = "gemini-2.5-flash-preview-tts"
    TTS_VOICE_NAME: str = "Despina"  # female voice, warm & expressive

    # Output
    OUTPUT_DIR: str = "./tmp/outputs"

    # Auth — shared secret with auth-service (must match JWT_SECRET_KEY there)
    JWT_SECRET_KEY: str = "your-super-secret-key-change-in-production"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance - call this everywhere instead of Settings()."""
    return Settings()