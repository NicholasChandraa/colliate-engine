import asyncio
from typing import cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from app.core.config import get_settings
from app.core.exceptions import StoryboardError
from app.core.logging import logger
from app.graph.state import GraphState
from app.schemas.storyboard import Storyboard

DIRECTOR_TIMEOUT_SECONDS = 180

def _build_llm(model: str) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the LangChain Google Generative AI model
    used for the Director node, configured with the appropriate API key and temperature.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.5,
        thinking_level=settings.THINKING_LEVEL,
    )

DIRECTOR_PROMPT = """
Kamu adalah seorang Creative Director berpengalaman untuk pembuatan video iklan komersial digital.

## DATA PRODUK/LAYANAN
{product_research}

## KONTEKS VISUAL
Tampilan fisik karakter dan produk sudah tersedia sebagai gambar referensi — JANGAN deskripsikan wajah, pakaian, atau bentuk produk secara spesifik di prompt.
Fokuskan semua instruksi visual pada: komposisi scene, posisi karakter, posisi produk, latar, dan pencahayaan.

## GLOBAL CONSISTENCY RULES
Aturan ini WAJIB dijaga konsistensinya di semua shot:
- Setting lokasi: Tentukan 1-2 lokasi utama yang relevan dan konsisten (misal: ruang tamu modern, outdoor cerah, meja kerja minimalis).
- Color grade & Mood: Tentukan nuansa warna dan pencahayaan keseluruhan (misal: warm & soft, bright & energetic, cinematic dark).
- Pastikan latar dan mood yang sama muncul di setiap `image_prompt`.

## LANGUAGE
Semua `video_prompt` dan `negative_prompt` WAJIB ditulis dalam bahasa Inggris — ini bahasa native Veo.
Ini penting agar Veo 3.1 memahami instruksi dengan akurat dan menghasilkan audio dalam bahasa yang benar.

## VIDEO PROMPT RULES (WAJIB DIIKUTI)
Veo adalah model generasi video berbasis fisika dan gerakan kamera — BUKAN video editor.
- JANGAN minta text overlay, teks animasi, atau tulisan di layar. Teks ditambahkan di post-production.
- JANGAN deskripsikan fisik/kondisi kulit karakter secara spesifik — ini dapat trigger safety filter.
- Struktur `video_prompt` harus mengikuti urutan: [Camera motion] + [Subject action] + [Style/ambiance] + [Audio cues].

Format audio dalam `video_prompt`:
- Dialogue: gunakan tanda kutip dengan acting cue. Contoh: "It works!" she said, smiling at the camera.
- SFX: deskripsikan eksplisit. Contoh: Soft click of a bottle cap opening, gentle pour.
- Ambient: deskripsikan soundscape. Contoh: Calm upbeat background music, light ambient noise.
- Jika karakter TIDAK berbicara pada shot tersebut (misal: close-up produk, texture shot): JANGAN tambahkan dialogue. Cukup tulis SFX dan/atau ambient yang sesuai. Contoh: No dialogue. Soft ASMR gel texture sound, calm background music.
- Akhiri selalu dengan: All audio and dialogue in {video_language}.

## TASK
Buat storyboard JSON untuk video iklan {target_duration} detik.
Pecah menjadi 8 shot.

Struktur shot yang direkomendasikan:
1. Hook (tarik perhatian) -> 2. Problem/Need -> 3. Product Reveal -> 4. Feature/Detail Close-up -> 5. Demonstration/Usage -> 6. Benefit/Result -> 7. Lifestyle/Happy Ending -> 8. CTA (Call to Action)

## OUTPUT FORMAT
Return ONLY valid JSON yang match schema ini - tanpa markdown, tanpa penjelasan:

{{
    "project_name": "string",
    "global_consistency": "string (deskripsi environment + color grade + mood yang konsisten di semua shot)",
    "shots": [
        {{
            "id": 1,
            "type": "hook|problem|solution|texture|application|result|lifestyle|cta",
            "camera_angle": "close-up|medium|wide",
            "camera_movement": "string",
            "subject_action": "string",
            "lighting": "string",
            "emotion": "string",
            "include_product": "boolean — true jika produk harus muncul di shot ini, false jika shot fokus ke karakter/emosi/lifestyle tanpa produk. Rekomendasi (tidak wajib): hook=false, problem=false, product_reveal=true, texture=true, application=true, result=false, lifestyle=false, cta=true.",
            "image_prompt": "string — Prompt untuk Imagen. Jika include_product=true: deskripsikan posisi karakter, posisi dan visibilitas produk, latar, pencahayaan. Jika include_product=false: fokus ke karakter, emosi, dan environment saja — jangan sebut produk. Jangan deskripsikan wajah atau tampilan fisik karakter. Format: static single frame, photorealistic.",
            "video_prompt": "string — IN ENGLISH. Veo prompt. Structure: [Camera motion]. [Subject action]. [Style/ambiance]. [Audio: dialogue in quotes + SFX + ambient]. End with: All audio and dialogue in {video_language}.",
            "negative_prompt": "string — IN ENGLISH. Elements to EXCLUDE. Describe what to avoid (NO 'no'/'don't'). Example: text overlay, watermarks, blurry motion, distorted hands, camera shake, explicit skin condition close-up."
        }}
    ]
}}
"""

async def director_node(state: GraphState) -> dict[str, list[dict[str, object]]]:
    """
    LangGraph Node: Acts as the Creative Director to formulate a cinematic storyboard.
    
    This node takes the 'product_research' generated by the research node and creates a structured, 
    JSON-formatted storyboard. It guarantees that global consistency rules (character, setting, mood) 
    are embedded into every single generation prompt ('image_prompt' and 'video_prompt').
    
    Args:
        state: The current state of the graph, containing the 'product_research' output.
        
    Returns:
        A dictionary containing the generated 'storyboard' list, where each 
        item is a parsed and validated shot configuration dictionary.
        
    Raises:
        StoryboardError: If the LLM call fails, or if Pydantic validation on the output JSON fails.
    """
    logger.info("🎬 Graph Node: LLM Director Started -> Generating Storyboard Plot")

    settings = get_settings()
    parser = JsonOutputParser()

    prompt = DIRECTOR_PROMPT.format(
        product_research=state.product_research,
        target_duration=settings.VIDEO_TARGET_DURATION_SECONDS,
        video_language=settings.VIDEO_LANGUAGE,
    )

    try:
        logger.info(f"📑 LLM PROMPT DIRECTOR: {prompt}")
        try:
            llm = _build_llm(settings.LLM_DIRECTOR_MODEL)
            llm_response = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=DIRECTOR_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"⏱️ Director LLM timed out after {DIRECTOR_TIMEOUT_SECONDS}s "
                f"— falling back to {settings.LLM_DIRECTOR_MODEL_FALLBACK}"
            )
            llm = _build_llm(settings.LLM_DIRECTOR_MODEL_FALLBACK)
            llm_response = await llm.ainvoke(prompt)
        logger.info(f"📑 LLM DIRECTOR RESPONSE: {llm_response}")
        
        content = llm_response.content
        if isinstance(content, str):
            json_string = content
        elif isinstance(content, list) and content and isinstance(content[0], dict):
            json_string = cast(str, content[0].get("text", ""))
        else:
            json_string = str(content)

        # Parse string to dictionary (LangChain parser handles markdown codeblocks)
        storyboard_dict = cast(dict[str, object], parser.parse(json_string))

        # Validate with Pydantic using model_validate (avoids kwargs unpacking type issues)
        storyboard = Storyboard.model_validate(storyboard_dict)

        logger.info(f"✅ LLM Director Completed -> Mastered {len(storyboard.shots)} shots.")
        return {"storyboard": [shot.model_dump() for shot in storyboard.shots]}

    except Exception as e:
        logger.error(f"❌ LLM Director Failed -> Error: {str(e)}")
        raise StoryboardError(f"Storyboard generation failed: {e}") from e