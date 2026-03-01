import uuid
from typing import Annotated
import traceback
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.logging import logger
from app.schemas.video import GenerativeVideoResponse
from app.services.job_service import create_job

router = APIRouter(prefix="/video", tags=["video"])

@router.post("/generate", response_model=GenerativeVideoResponse, status_code=202)
async def generate_video(
    product_name: Annotated[str, Form(...)],
    target_audience: Annotated[str, Form(...)],
    character_image: Annotated[UploadFile, File(...)],
    product_image: Annotated[UploadFile, File(...)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
) -> GenerativeVideoResponse:
    f"""
    Enqueue a video generation job.
    Returns immediately with a job_id - poll /jobs/:job_id to trach progress.
    """

    job_id = str(uuid.uuid4())
    logger.info(f"📥 Received Request: Generate Video -> Job: [{job_id}], Product: '{product_name}'")

    if not character_image.content_type or not character_image.content_type.startswith("image/"):
        logger.warning(f"⚠️ Rejected Request: Invalid character_image format -> {character_image.content_type}")
        raise HTTPException(status_code=400, detail="character_image must be an image")
    
    if not product_image.content_type or not product_image.content_type.startswith("image/"):
        logger.warning(f"⚠️ Rejected Request: Invalid product_image format -> {product_image.content_type}")
        raise HTTPException(status_code=400, detail="product_image must be an image")

    try:
        character_bytes = await character_image.read()
        product_bytes = await product_image.read()

        # Persist job record to DB
        _ = await create_job(
            db=db,
            job_id=job_id,
            product_name=product_name,
            target_audience=target_audience
        )
    except Exception as e:
        logger.error(f"❌ Failed to process base files / save to DB [Job: {job_id}] -> Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while initializing job")

    try:
        # Enqueue background task to ARQ
        settings = get_settings()
        redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))

        _ = await redis.enqueue_job(
            "generate_video_task",
            job_id=job_id,
            product_name=product_name,
            target_audience=target_audience,
            character_image_bytes=character_bytes,
            product_image_bytes=product_bytes,
            _job_id=job_id    # ARQ job ID = job ID untuk traceability
        )

        await redis.close()
        logger.info(f"📤 Successfully Dispatched to Redis Background Worker -> Job: [{job_id}]")

    except Exception as e:
        # Jika Redis gagal, kita harus mencatatnya dan memberi tahu front-end
        logger.error(f"❌ Failed to enqueue task to Redis [Job: {job_id}] -> Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail="Service unavailable, failed to queue job")

    return GenerativeVideoResponse(
        job_id=job_id,
        status="pending",
        message=f"Job enqueued. Poll /api/jobs/{job_id} to track progress."
    )