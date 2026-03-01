from pydantic import BaseModel, Field

class GraphState(BaseModel):
    """
    Immutable-style state that flows through the LangGraph pipeline.
    Each node returns partial dict - Langgraph merges it into the state.
    """

    # Inputs
    job_id: str = Field(default="")
    product_name: str = Field(default="")
    target_audience: str = Field(default="")
    character_image_bytes: bytes = Field(default=b"")
    product_image_bytes: bytes = Field(default=b"")

    # Stage: Research
    product_research: str = Field(default="")

    # Stage: Director
    # Stored as raw list[dict] so it stays JSON-serializable in LangGraph
    storyboard: list[dict[str, object]] = Field(default_factory=list)

    # Stage: Execution
    generated_video_paths: list[str] = Field(default_factory=list)

    # Stage: Assembly
    final_video_path: str = Field(default="")
    
    # Error handling
    error: str | None = Field(default=None)