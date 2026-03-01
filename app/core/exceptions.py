class VideoAdGeneratorError(Exception):
    """
    Base exception for all application errors.
    """

class ResearchError(VideoAdGeneratorError):
    """
    Raised when product research fails.
    """

class StoryboardError(VideoAdGeneratorError):
    """
    Raised when storyboard generation fails.
    """

class ImageGenerationError(VideoAdGeneratorError):
    """Raised when scene image generation fails."""


class VideoGenerationError(VideoAdGeneratorError):
    """Raised when video clip generation fails."""

class VideoSafetyFilterError(VideoGenerationError):
    """Raised when Veo blocks generation due to safety filter. Not retried — prompt won't change."""

class VideoRateLimitError(VideoGenerationError):
    """Raised when Veo returns 429 RESOURCE_EXHAUSTED. Retried with a longer wait."""


class AssemblyError(VideoAdGeneratorError):
    """Raised when FFmpeg assembly fails."""
