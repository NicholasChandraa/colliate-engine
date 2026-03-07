import uuid
from typing_extensions import override
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import DateTime, Enum, Integer, String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class JobStatus(str, PyEnum):
    PENDING = "pending"
    RESEARCHING = "researching"
    DIRECTING = "directing"
    GENERATING_IMAGES = "generating_images"   # Task 1: generating 2 images per shot
    AWAITING_SELECTION = "awaiting_selection" # Task 1 done, waiting for user to select images
    GENERATING_VIDEOS = "generating_videos"   # Task 2: generating video clips for selected shots
    ASSEMBLING = "assembling"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"  # product failed safety/quality review

class Job(Base):
    __tablename__: str = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    product_research: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), nullable=False, default=JobStatus.PENDING
    )

    # Progress tracking - e.g. "Shot 3/8 generating..."
    progress_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Shot progress counters
    total_shots: Mapped[int] = mapped_column(Integer, default=0)
    completed_shots: Mapped[int] = mapped_column(Integer, default=0)

    # Assets
    product_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_image_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "product" | "character" | "skin"
    final_video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now, nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.now, onupdate=datetime.now, nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @override
    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status}>"

    # Relationships
    shots: Mapped[list["JobShot"]] = relationship("JobShot", back_populates="job", cascade="all, delete-orphan", order_by="JobShot.shot_index")


class JobShotStatus(str, PyEnum):
    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


class JobShot(Base):
    __tablename__: str = "job_shots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    
    shot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    shot_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Prompts & Metadata
    camera_angle: Mapped[str | None] = mapped_column(String(100), nullable=True)
    camera_movement: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subject_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    lighting: Mapped[str | None] = mapped_column(String(255), nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(100), nullable=True)
    voiceover_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    image_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Generated Assets — 2 image options per shot, user selects one
    scene_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)   # option 1
    scene_image_path_2: Mapped[str | None] = mapped_column(Text, nullable=True) # option 2
    selected_image: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 or 2
    audio_clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)    # TTS .wav output
    video_clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Progress
    status: Mapped[JobShotStatus] = mapped_column(Enum(JobShotStatus), nullable=False, default=JobShotStatus.PENDING)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now, nullable=True)
    
    # Relationships
    job: Mapped["Job"] = relationship("Job", back_populates="shots")

    @override
    def __repr__(self) -> str:
        return f"<JobShot id={self.id} job_id={self.job_id} index={self.shot_index} status={self.status}>"

        