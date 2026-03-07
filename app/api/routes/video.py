import uuid
import os
from pathlib import Path
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
from app.api.middleware.auth import get_optional_user_id

router = APIRouter(prefix="/video", tags=["video"])

@router.post("/generate", response_model=GenerativeVideoResponse, status_code=202)
async def generate_video(
    product_name: Annotated[str, Form(...)],
    product_image: Annotated[UploadFile, File(...)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    title: Annotated[str | None, Form()] = None,
    user_id: Annotated[str | None, Depends(get_optional_user_id)] = None,
    reference_image: Annotated[UploadFile | None, File()] = None,
    reference_image_type: Annotated[str | None, Form()] = None,
) -> GenerativeVideoResponse:
    f"""
    Enqueue a skincare science video generation job.
    Returns immediately with a job_id - poll /jobs/:job_id to track progress.
    The research node will evaluate product safety before proceeding.
    """

    job_id = str(uuid.uuid4())
    logger.info(f"📥 Received Request: Generate Video -> Job: [{job_id}], Product: '{product_name}'")

    if not product_image.content_type or not product_image.content_type.startswith("image/"):
        logger.warning(f"⚠️ Rejected Request: Invalid product_image format -> {product_image.content_type}")
        raise HTTPException(status_code=400, detail="product_image must be an image")

    try:
        product_bytes = await product_image.read()

        # Save product image to disk for persistent reference
        settings = get_settings()
        ext = (product_image.filename or "product.jpg").rsplit(".", 1)[-1].lower() or "jpg"
        job_dir = Path(settings.OUTPUT_DIR) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        product_image_path = str(job_dir / f"product_image.{ext}")
        with open(product_image_path, "wb") as f:
            f.write(product_bytes)

        # Save optional reference image (extra product angle / character / skin)
        reference_bytes: bytes = b""
        ref_image_path: str | None = None
        ref_type: str | None = reference_image_type if reference_image_type in ("product", "character", "skin") else None
        if reference_image and reference_image.content_type and reference_image.content_type.startswith("image/"):
            reference_bytes = await reference_image.read()
            ref_ext = (reference_image.filename or "reference.jpg").rsplit(".", 1)[-1].lower() or "jpg"
            ref_image_path = str(job_dir / f"reference_image.{ref_ext}")
            with open(ref_image_path, "wb") as f:
                f.write(reference_bytes)

        _ = await create_job(
            db=db,
            job_id=job_id,
            title=title,
            product_name=product_name,
            user_id=user_id,
            product_image_path=product_image_path,
            reference_image_path=ref_image_path,
            reference_image_type=ref_type,
        )
    except Exception as e:
        logger.error(f"❌ Failed to process files / save to DB [Job: {job_id}] -> Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while initializing job")

    try:
        settings = get_settings()
        redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))

        _ = await redis.enqueue_job(
            "generate_images_task",
            job_id=job_id,
            product_name=product_name,
            product_image_bytes=product_bytes,
            reference_image_bytes=reference_bytes,
            reference_image_type=ref_type or "",
            _job_id=job_id
        )

        await redis.close()
        logger.info(f"📤 Successfully Dispatched to Redis Background Worker -> Job: [{job_id}]")

    except Exception as e:
        logger.error(f"❌ Failed to enqueue task to Redis [Job: {job_id}] -> Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail="Service unavailable, failed to queue job")

    return GenerativeVideoResponse(
        job_id=job_id,
        status="pending",
        message=f"Job enqueued. Poll /api/jobs/{job_id} to track progress."
    )
