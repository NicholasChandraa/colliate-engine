import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import video_router, job_router
from app.core.config import get_settings

app = FastAPI(
    title="Video Ad Generator",
    description="AI-Powered long-form video ad generator using LangGraph + Google Veo 3.1",
    version="0.1.0",
)

# ── CORS ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan origin spesifik di tahap produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ─────────────────────────────────────────────────────────
app.include_router(video_router, prefix="/api")
app.include_router(job_router, prefix="/api")

# ── Static Files (Serving Generative Videos) ────────────────────────────
settings = get_settings()
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/api/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    reload = True if settings.APP_ENV == "development" else False
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload)