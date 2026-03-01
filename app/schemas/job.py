from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import ClassVar, Sequence
from app.models.job import JobStatus, JobShotStatus

class JobShotSchema(BaseModel):
    id: str
    shot_index: int
    shot_type: str
    camera_angle: str | None = None
    camera_movement: str | None = None
    subject_action: str | None = None
    lighting: str | None = None
    emotion: str | None = None
    voiceover_text: str | None = None
    image_prompt: str | None = None
    video_prompt: str | None = None
    scene_image_path: str | None = None
    video_clip_path: str | None = None
    status: JobShotStatus
    error_message: str | None = None
    
    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    product_research: str | None = None
    progress_message: str | None = None
    total_shots: int = 0
    completed_shots: int = 0
    shots: Sequence[JobShotSchema] = []
    final_video_path: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)