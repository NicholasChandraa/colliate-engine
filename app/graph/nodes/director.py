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
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.5,
        thinking_level=settings.THINKING_LEVEL,
    )


DIRECTOR_PROMPT = """
Kamu adalah Kurator Kecantikan Klinis yang merancang konten video edukasi skincare berbasis sains.

Persona: Logis, edukatif, to-the-point. Bicara seperti dokter kulit yang tahu cara bercerita.
Bukan influencer beauty — tidak ada hype, tidak ada pujian berlebihan. Fokus pada problem-solving.

## DATA PRODUK (HASIL RISET ILMIAH)
{product_research}

## RINGKASAN KUNCI PRODUK INI (WAJIB DIGUNAKAN)
Bahan aktif utama: {key_ingredients}
Masalah kulit yang ditarget: {skin_concerns}

WAJIB: Seluruh storyboard harus SPESIFIK untuk bahan aktif dan masalah kulit di atas.
- Shot ingredient (3 & 4) HARUS menampilkan bahan aktif dari list di atas — BUKAN bahan generik lain.
- Shot kulit (1, 2, 7) HARUS menampilkan gejala visual dari masalah kulit di atas — BUKAN masalah kulit yang tidak relevan.
- Voiceover HARUS menyebut manfaat spesifik produk ini — BUKAN manfaat skincare generik.

## ATURAN VISUAL — WAJIB DIIKUTI
1. DILARANG: Wajah manusia realistis. Tubuh manusia realistis.
2. GAYA VISUAL: 100% photorealistic photography — macro, still life, ingredient, formula.
   Tidak ada ilustrasi saintifik, tidak ada diagram, tidak ada CGI cartoon/stylized.
   Referensi estetik: kampanye ingredient Dior/Lancôme, editorial majalah Vogue Beauty,
   iklan serum L'Oreal dengan close-up tetesan serum.
3. Produk HANYA muncul di shot 6 dan 8 (include_product: true). Shot lainnya: false.
4. Untuk shot produk: JANGAN sebutkan warna botol, tutup, label, atau bentuk produk.
   Cukup deskripsikan background, lighting, props, dan komposisi saja.

## VISUAL LITERAL PER TIPE SHOT

HOOK (shot 1) — Masalah kulit spesifik produk ini, macro realistis:
→ Extreme macro photography {skin_area} — bukan foto wajah utuh, hanya tekstur permukaan.
  WAJIB tulis "{skin_area}, young adult skin age 25-35, macro" secara eksplisit di image_prompt.
  WAJIB tampilkan masalah kulit dari list SKIN CONCERNS produk ini — bukan masalah kulit generik.
  Contoh: jika produk untuk brightening → tampilkan hiperpigmentasi/dark spots. Jika untuk barrier → tampilkan kulit kering/kemerahan/tekstur kasar.
  Seperti foto mikroskop kamera DSLR, bukan ilustrasi.
  Lighting: dramatic side lighting untuk menonjolkan tekstur.

AGITATION (shot 2) — Masalah lebih detail, macro lebih dekat:
→ Zoom in lebih ekstrem dari shot 1. Tetap gunakan area yang sama: {skin_area}.
  WAJIB tulis "{skin_area}, young adult skin age 25-35, extreme macro" di image_prompt.
  WAJIB fokus pada SATU gejala visual spesifik dari skin concern produk ini.
  Depth of field sangat sempit. Background soft bokeh.

INGREDIENT_ORIGIN (shot 3) — Sumber alami bahan aktif UTAMA produk ini:
→ Tentukan sendiri sumber alami dari bahan aktif yang ada di list RINGKASAN KUNCI di atas.
  Pilih bahan aktif yang paling iconic/kuat dari produk ini.
  Visualisasikan sumber alaminya secara macro — segar, alami, premium.
  Seperti food photography editorial atau nature macro photography.
  WAJIB: image_prompt harus menyebut nama bahan aktif spesifik produk ini.

INGREDIENT_EXTRACTED (shot 4) — Bentuk murni bahan aktif tersebut:
→ Visualisasikan bahan aktif yang sama dari shot 3 dalam bentuk yang sudah diekstrak/dimurnikan.
  Tentukan form visualnya berdasarkan sifat bahan aktif tersebut:
  - Bahan berbentuk kristal/powder: butiran/kristal di permukaan gelap, soft spotlight dari atas
  - Bahan berbentuk cairan/serum: tetesan dari pipette kaca, cairan bening berkilau
  - Bahan berbentuk gel/emulsi: permukaan gel smooth, refleksi cahaya di atasnya
  WAJIB: image_prompt harus menyebut nama bahan aktif spesifik dan bentuk visualnya secara eksplisit.

FORMULA_SENSORIAL (shot 5) — Formula produk sebagai seni:
→ Macro photography formula/tekstur produk itu sendiri (bukan kemasannya).
  WARNA FORMULA YANG BENAR: {formula_color} — gunakan warna ini secara eksplisit di image_prompt.
  TEKSTUR: {formula_texture} — tentukan visual berdasarkan tekstur ini.
  Contoh visual sesuai tekstur:
  - lightweight gel / gel: gel ditarik perlahan, tekstur elastis terlihat, atau diteteskan ke permukaan kaca
  - watery serum / serum: tetesan jatuh ke permukaan kaca → ripple effect, gelembung mikro
  - thick cream / krim: krim dioleskan ke kaca transparan, meninggalkan lapisan halus
  - milky lotion: aliran lotion putih susu dituang perlahan
  Sangat sensorial, mengundang. Seperti iklan parfum atau serum high-end.

PRODUCT_REVEAL (shot 6) — include_product: TRUE:
→ Deskripsikan hanya: background, lighting, komposisi.
  Contoh: "Clean white marble surface, soft top-down studio lighting, minimal composition."

RESULT (shot 7) — Kulit sehat, macro realistis:
→ Macro photography {skin_area} yang sehat sebagai kontras dari shot 1.
  WAJIB tulis "{skin_area}, young adult skin age 25-35, macro" di image_prompt — area dan usia harus sama dengan shot 1 dan 2.
  Tekstur halus, pori kecil, tone merata, ada sedikit moisture glow di permukaan.
  Bukan wajah — hanya tekstur permukaan kulit yang sehat dan bercahaya.
  Lighting: soft, flattering, warm glow.

CTA (shot 8) — include_product: TRUE:
→ Deskripsikan hanya: background, props estetik, lighting, suasana.
  Contoh: "Soft pink background, small botanical props, warm diffused light, lifestyle feel."

## ATURAN HOOK — SHOT PERTAMA
Hook WAJIB menggunakan formula: [Pain Point Spesifik] + [Kesalahan Umum yang Dilakukan Orang].
DILARANG: "Ini rekomendasi serum terbaik..." atau kalimat yang sekadar memuji produk.
HARUS: "Rutin 7 langkah tapi kulit masih kusam? Itu karena urutan aplikasinya salah."
         "Jerawat batu nggak hilang padahal udah cuci muka 3x sehari? Itu karena kamu merusak skin barrier."

## LANGUAGE & AUDIO
Semua `video_prompt` dan `negative_prompt` WAJIB ditulis dalam bahasa Inggris.
Struktur `video_prompt`: [Camera motion]. [Visual subject]. [Style/ambiance]. [Audio: SFX + ambient music ONLY].

Audio format untuk `video_prompt`:
- WAJIB: Hanya SFX dan ambient music. Contoh: "Soft molecular zoom SFX, calm clinical background music."
- DILARANG: Voiceover, dialog, Female voiceover, male voice, atau instruksi suara manusia APAPUN.
  Voiceover akan ditambahkan terpisah via TTS — JANGAN masukkan ke video_prompt.
- Akhiri selalu dengan: "No dialogue, no voiceover."

Field `voiceover_text` (per shot, bahasa Indonesia):
- Narasi yang akan di-synthesize oleh TTS secara terpisah
- Tulis seperti dokter kulit yang berbicara langsung, klinis tapi hangat
- Maks 2 kalimat pendek. Tidak boleh lebih dari 30 kata.
- Contoh hook: "Rutin skincare-mu sudah benar, tapi urutan aplikasinya yang membuat kulit kusam."
- Contoh cta: "Coba sekarang dan rasakan perbedaannya dalam 14 hari."

## TASK
Buat storyboard JSON untuk video edukasi skincare {target_duration} detik.
Pecah menjadi tepat 8 shot dengan struktur berikut:

| Shot | Type                 | include_product | Visual                                          |
|------|----------------------|-----------------|-------------------------------------------------|
| 1    | hook                 | false           | Macro kulit bermasalah — tekstur realistis      |
| 2    | agitation            | false           | Macro lebih ekstrem — detail masalah kulit      |
| 3    | ingredient_origin    | false           | Sumber alami bahan aktif — food/nature macro    |
| 4    | ingredient_extracted | false           | Bahan aktif murni — kristal, bubuk, atau cairan |
| 5    | formula_sensorial    | false           | Formula produk sebagai seni — tetes, tekstur    |
| 6    | product_reveal       | TRUE            | Product studio shot — pertama kali muncul       |
| 7    | result               | false           | Macro kulit sehat — kontras dari shot 1         |
| 8    | cta                  | TRUE            | Aesthetic lifestyle product shot                |

## GLOBAL CONSISTENCY
Tentukan satu visual language photorealistic yang konsisten:
- Color palette: pilih warm (krem, gold, blush) atau cool (putih, mint, silver) — konsisten
- Lighting style: pilih satu (soft diffused studio / dramatic side lighting / warm backlight)
- Photography aesthetic: pilih satu (clean minimal / nature editorial / clinical premium)
- Skin age (WAJIB konsisten di shot 1, 2, 7): gunakan "young adult skin, age 25-35" — tulis eksplisit di setiap image_prompt skin shot. DILARANG menggunakan kata "aged", "elderly", "wrinkled", "mature skin" kecuali produk memang spesifik untuk anti-aging.
Konsistensi ini WAJIB disebut di setiap `image_prompt`.

## OUTPUT FORMAT
Return ONLY valid JSON — tanpa markdown, tanpa penjelasan, tanpa teks lain:

{{
    "project_name": "string",
    "global_consistency": "string — scientific color palette + lighting style + render style yang konsisten",
    "shots": [
        {{
            "id": 1,
            "type": "hook|agitation|ingredient_origin|ingredient_extracted|formula_sensorial|product_reveal|result|cta",
            "camera_angle": "extreme-close-up|close-up|medium|wide",
            "camera_movement": "string",
            "subject_action": "string — deskripsi apa yang divisualisasikan (bukan karakter manusia)",
            "lighting": "string",
            "emotion": "string — tone/feel shot ini: e.g. 'clinical', 'dramatic', 'hopeful', 'scientific', 'empowering'",
            "include_product": false,
            "image_prompt": "string — Prompt untuk Imagen. Scientific visualization / macro / product mockup. DILARANG menyebut wajah atau tubuh manusia. Untuk shot include_product=true: HANYA deskripsikan scene composition, background, dan lighting — JANGAN sebutkan warna/label/bentuk produk. Format: static single frame.",
            "video_prompt": "string — IN ENGLISH. [Camera motion]. [Visual]. [Style]. [Audio: SFX + ambient music ONLY, NO voiceover, NO dialogue]. End: 'No dialogue, no voiceover.'",
            "negative_prompt": "string — IN ENGLISH. realistic human faces, human body parts, text overlay, watermarks, blurry motion, distorted shapes, camera shake.",
            "voiceover_text": "string — BAHASA INDONESIA. Narasi TTS 1-2 kalimat maks 30 kata. Klinis, edukatif, di-generate terpisah via Gemini TTS."
        }}
    ]
}}
"""


async def director_node(state: GraphState) -> dict[str, list[dict[str, object]]]:
    """
    LangGraph Node: Clinical Beauty Curator — generates a science-based educational storyboard.

    Takes the product scientific brief from the research node and creates an 8-shot
    educational storyboard focused on skin science visualization with soft-selling
    (product only revealed at shots 6 and 8).

    Raises:
        StoryboardError: If the LLM call fails or Pydantic validation fails.
    """
    logger.info("🎬 Graph Node: Clinical Director Started -> Generating Science Storyboard")

    settings = get_settings()
    parser = JsonOutputParser()

    prompt = DIRECTOR_PROMPT.format(
        product_research=state.product_research,
        target_duration=settings.VIDEO_TARGET_DURATION_SECONDS,
        formula_color=state.formula_color or "translucent",
        formula_texture=state.formula_texture or "gel",
        skin_area=state.skin_area or "cheek area skin surface",
        key_ingredients=", ".join(state.key_ingredients) if state.key_ingredients else "lihat hasil riset",
        skin_concerns=", ".join(state.skin_concerns) if state.skin_concerns else "lihat hasil riset",
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

        storyboard_dict = cast(dict[str, object], parser.parse(json_string))
        storyboard = Storyboard.model_validate(storyboard_dict)

        logger.info(f"✅ Clinical Director Completed -> Mastered {len(storyboard.shots)} shots.")
        return {"storyboard": [shot.model_dump() for shot in storyboard.shots]}

    except Exception as e:
        logger.error(f"❌ Clinical Director Failed -> Error: {str(e)}")
        raise StoryboardError(f"Storyboard generation failed: {e}") from e
