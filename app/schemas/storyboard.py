from pydantic import BaseModel, Field

class Shot(BaseModel):
    id: int = Field(..., description="Shot sequence number, starts at 1")
    type: str = Field(..., description="e.g. 'hook', 'problem', 'solution', 'detail', 'cta'")
    camera_angle: str = Field(..., description="e.g. 'close-up', 'medium', 'wide'")
    camera_movement: str = Field(..., description="e.g. 'slow push in', 'static', 'pan left'")
    subject_action: str = Field(..., description="What the character does in this shot")
    lighting: str = Field(..., description="e.g. 'soft natural light from right, warm'")
    emotion: str = Field(..., description="e.g. 'confident', 'curious', 'joyful'")
    include_product: bool = Field(..., description="Whether the product should appear in this shot")
    image_prompt: str = Field(..., description="Static photorealistic description for initial scene")
    video_prompt: str = Field(..., description="Cinematic physics/motion description for Veo 3.1. SFX and ambient music only — NO voiceover, NO dialogue.")
    negative_prompt: str = Field(..., description="Elements to exclude from the video (no 'no/don't', just describe)")
    voiceover_text: str = Field(..., description="Script voiceover bahasa Indonesia untuk TTS. Maks 2 kalimat pendek, tone klinis & edukatif.")


class Storyboard(BaseModel):
    project_name: str
    global_consistency: str = Field(
        ...,
        description="Character + environment description enforced across ALL shots"
    )
    shots: list[Shot]