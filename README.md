# Video Ad Generator

AI-powered long-form video ad generator (30–60 detik) menggunakan LangGraph + Google Veo 3.1.

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM / Image Gen | LangChain + `langchain-google-genai` |
| Video Gen | Google GenAI SDK (`google-genai`) |
| API | FastAPI |
| Audio | FFmpeg |

## Models

| Role | Model |
|---|---|
| Research | `gemini-3-flash-preview` |
| Director (Storyboard) | `gemini-3.1-pro-preview` |
| Image Gen (Nano Banana 2) | `gemini-3.1-flash-image-preview` |
| Video Gen (Veo 3.1) | `veo-3.1-generate-preview` |
| TTS | `gemini-2.5-flash-preview-tts` |

## Setup

```bash
# 1. Clone & masuk folder
git clone <repo>
cd video_ad_generator

# 2. Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env → isi GOOGLE_API_KEY

# 5. Pastikan FFmpeg terinstall
ffmpeg -version

# 6. Jalankan server
uvicorn main:app --reload
```

## Project Structure

```
video_ad_generator/
├── main.py                     # FastAPI entry point
├── src/
│   ├── api/
│   │   └── routes/
│   │       └── video.py        # POST /api/v1/video/generate
│   ├── core/
│   │   ├── config.py           # Centralized settings (pydantic-settings)
│   │   ├── exceptions.py       # Custom exception hierarchy
│   │   └── logging.py          # Structured logging (structlog)
│   ├── graph/
│   │   ├── state.py            # LangGraph GraphState
│   │   ├── graph.py            # Pipeline definition
│   │   └── nodes/
│   │       ├── research.py     # Node 1: Product research
│   │       ├── director.py     # Node 2: Storyboard generation
│   │       ├── shot_loop.py    # Node 3: Image + Video generation per shot
│   │       └── assembly.py     # Node 4: FFmpeg stitching
│   ├── schemas/
│   │   ├── storyboard.py       # Shot & Storyboard Pydantic models
│   │   └── video.py            # API request/response schemas
│   └── services/
│       └── frame_extractor.py  # FFmpeg last-frame extraction utility
├── tests/
├── .env.example
├── requirements.txt
└── pyproject.toml
```

## API Usage

```bash
curl -X POST http://localhost:8000/api/v1/video/generate \
  -F "product_name=Skintific MSH Niacinamide" \
  -F "target_audience=Wanita 18-35th, masalah kulit kusam" \
  -F "character_image=@/path/to/character.jpg" \
  -F "product_image=@/path/to/product.jpg"
```