from pydantic import BaseModel, Field

class GraphState(BaseModel):
    """
    Immutable-style state that flows through the LangGraph pipeline.
    Each node returns partial dict - Langgraph merges it into the state.
    """

    # Inputs
    job_id: str = Field(default="")
    product_name: str = Field(default="")
    product_image_bytes: bytes = Field(default=b"")
    reference_image_bytes: bytes = Field(default=b"")   # optional: extra product angle / character / skin
    reference_image_type: str = Field(default="")       # "product" | "character" | "skin"

    # Stage: Research
    product_research: str = Field(default="")
    product_verdict: str = Field(default="")          # "APPROVED" | "REJECTED"
    product_verdict_reason: str = Field(default="")
    formula_color: str = Field(default="")            # e.g. "white", "pale yellow"
    formula_texture: str = Field(default="")          # e.g. "gel", "serum", "cream"
    skin_area: str = Field(default="")               # e.g. "cheek area skin surface", "forearm skin surface"
    key_ingredients: list[str] = Field(default_factory=list)   # e.g. ["Niacinamide 5%", "Ceramide NP"]
    skin_concerns: list[str] = Field(default_factory=list)     # e.g. ["dark spots", "uneven tone"]

    # Stage: Director
    # Stored as raw list[dict] so it stays JSON-serializable in LangGraph
    storyboard: list[dict[str, object]] = Field(default_factory=list)

    # Stage: Execution
    generated_video_paths: list[str] = Field(default_factory=list)

    # Stage: Assembly
    final_video_path: str = Field(default="")

    # Error handling
    error: str | None = Field(default=None)
