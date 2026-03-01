from app.api.routes.jobs import router as job_router
from app.api.routes.video import router as video_router

__all__ = [
    "job_router",
    "video_router"
]