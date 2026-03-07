from pydantic import BaseModel, Field

class GenerateVideoRequest(BaseModel):
    product_name: str = Field(..., min_length=2, description="e.g. 'Skintific MSH Niacinamide Barrier Serum'")

class GenerativeVideoResponse(BaseModel):
    job_id: str
    status: str
    final_video_path: str | None = None
    message: str | None = None
