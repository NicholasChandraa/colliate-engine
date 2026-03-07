from typing import cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from app.core.config import get_settings
from app.core.exceptions import ResearchError
from app.core.logging import logger
from app.graph.state import GraphState


def _build_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.LLM_RESEARCH_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )


RESEARCH_PROMPT = """
Kamu adalah Skincare Science Analyst yang bertugas mengevaluasi produk skincare secara ilmiah dan objektif.
Fokusmu adalah pada bukti saintifik — bukan marketing atau hype.

Produk: {product_name}

Gunakan Google Search untuk mencari informasi terkini dan akurat.
Cari: komposisi bahan, studi klinis, ulasan dermatologis, laporan keamanan, dan kontroversi jika ada.

Kembalikan brief ilmiah dalam format berikut (bahasa Indonesia):

## PROFIL PRODUK
Nama lengkap, kategori produk, dan klaim utama yang dibuat produsen.

## KANDUNGAN AKTIF & MEKANISME KERJA
Untuk setiap bahan aktif utama:
- Nama bahan aktif
- Mekanisme kerja berdasarkan penelitian ilmiah
- Level evidence (terbukti kuat / menjanjikan / terbatas / anekdotal)

## MASALAH KULIT YANG DISELESAIKAN
Kondisi kulit spesifik yang ditarget beserta penjelasan patofisiologi singkatnya.
Contoh: jerawat → overproduction sebum + kolonisasi C. acnes → inflamasi pori.

## KEAMANAN & KONTRAINDIKASI
Profil keamanan bahan aktif. Adakah iritan, alergen umum, atau kontraindikasi?
Kelompok kulit yang perlu berhati-hati (kulit sensitif, bumil, dll).

## FORMULA & TEKSTUR
Deskripsikan penampilan fisik formula produk (bukan kemasan):
- Warna formula: warna asli formula/gel/serum/krim itu sendiri (bukan warna kemasan/botol)
  Contoh: "bening", "putih susu", "kuning pucat", "hijau muda"
- Tekstur: jenis tekstur formula
  Contoh: "gel ringan", "serum cair", "krim kental", "lotion ringan"
Penting: warna kemasan/botol sering berbeda dengan warna formula itu sendiri.

## KESIMPULAN ILMIAH
Ringkasan: apakah klaim produk didukung bukti ilmiah? Apakah formulasinya masuk akal?
"""

VERDICT_PROMPT = """
Berdasarkan hasil riset ilmiah produk skincare berikut, buat keputusan apakah produk ini
layak untuk konten edukasi soft-selling kepada konsumen Indonesia.

HASIL RISET:
{product_research}

Kriteria APPROVED:
- Bahan aktif terbukti secara ilmiah (minimal ada peer-reviewed evidence)
- Klaim produk masuk akal dan tidak berlebihan
- Tidak ada bahan yang berpotensi berbahaya atau kontraindikasi serius

Kriteria REJECTED:
- Klaim menyesatkan atau tidak didukung bukti ilmiah
- Mengandung bahan bermasalah (iritan tinggi, bahan dilarang, berbahaya)
- Produk tidak jelas identitasnya atau tidak bisa diverifikasi sama sekali

Return ONLY valid JSON (tanpa markdown, tanpa penjelasan tambahan):
{{
    "verdict": "APPROVED atau REJECTED",
    "reason": "Penjelasan 1-2 kalimat yang jelas mengapa produk ini diapprove atau direjeksi",
    "key_ingredients": ["bahan aktif 1", "bahan aktif 2"],
    "skin_concerns": ["masalah kulit 1", "masalah kulit 2"],
    "formula_color": "warna asli formula itu sendiri dalam bahasa Inggris, bukan warna kemasan. Contoh: 'white', 'translucent', 'pale yellow', 'light green'",
    "formula_texture": "jenis tekstur formula dalam bahasa Inggris. Contoh: 'lightweight gel', 'watery serum', 'thick cream', 'milky lotion'",
    "skin_area": "area kulit spesifik yang relevan untuk produk ini, dalam bahasa Inggris, untuk dipakai sebagai subjek foto macro. Pilih yang paling tepat berdasarkan fungsi produk: 'cheek area skin surface' (produk wajah), 'forearm skin surface' (produk tubuh/body lotion), 'scalp skin surface' (produk rambut/kulit kepala), 'under-eye skin surface' (produk mata), 'neck and décolleté skin surface' (produk leher). Hanya kembalikan string area-nya saja."
}}
"""


async def research_node(state: GraphState) -> dict[str, str]:
    """
    LangGraph Node: Skincare Science Analyst.

    Call 1 (Google Search grounding): Researches the product scientifically —
    ingredients, mechanisms, evidence level, safety profile.

    Call 2 (JSON verdict): Based on the research, decides APPROVED or REJECTED.
    Returns both the scientific brief (for the director node) and the verdict.

    Returns:
        dict with keys: product_research, product_verdict, product_verdict_reason

    Raises:
        ResearchError: If any LLM call fails or returns empty content.
    """
    logger.info(f"🔬 Graph Node: Skincare Science Analyst Started -> Researching '{state.product_name}'")

    # ── Call 1: Google Search grounding — scientific brief ────────────────
    llm = _build_llm().bind_tools([{"google_search": {}}])
    prompt = RESEARCH_PROMPT.format(product_name=state.product_name)

    try:
        logger.info(f"📑 LLM PROMPT RESEARCHER: {prompt}")
        research_response = await llm.ainvoke(prompt)
        logger.info(f"📑 LLM RESEARCHER RESPONSE: {research_response}")

        grounding = research_response.response_metadata.get("grounding_metadata") if research_response.response_metadata else None
        if grounding:
            search_queries = grounding.get("web_search_queries", [])
            logger.info(f"🔍 Google Search Used -> Queries: {search_queries}")
        else:
            logger.warning("⚠️ Google Search NOT used — response has no grounding_metadata")

        raw = research_response.content
        if isinstance(raw, str):
            scientific_brief = raw
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            scientific_brief = cast(str, raw[0].get("text", ""))
        else:
            scientific_brief = str(raw)

        if not scientific_brief.strip():
            raise ResearchError("Research LLM returned empty content")

        logger.info(f"✅ Scientific Brief Gathered -> {len(scientific_brief)} characters")

    except ResearchError:
        raise
    except Exception as e:
        logger.error(f"❌ Research Call Failed -> Error: {str(e)}")
        raise ResearchError(f"Research failed: {e}") from e

    # ── Call 2: Verdict — no grounding needed ────────────────────────────
    verdict_llm = _build_llm()
    parser = JsonOutputParser()
    verdict_prompt = VERDICT_PROMPT.format(product_research=scientific_brief)

    try:
        logger.info("⚖️ Running product verdict evaluation...")
        verdict_response = await verdict_llm.ainvoke(verdict_prompt)

        verdict_raw = verdict_response.content
        if isinstance(verdict_raw, list) and verdict_raw and isinstance(verdict_raw[0], dict):
            verdict_str = cast(str, verdict_raw[0].get("text", ""))
        else:
            verdict_str = str(verdict_raw)

        verdict_data = cast(dict[str, object], parser.parse(verdict_str))
        verdict = str(verdict_data.get("verdict", "REJECTED")).upper()
        reason = str(verdict_data.get("reason", "Tidak ada alasan diberikan."))
        formula_color = str(verdict_data.get("formula_color", "translucent"))
        formula_texture = str(verdict_data.get("formula_texture", "gel"))
        skin_area = str(verdict_data.get("skin_area", "cheek area skin surface"))
        key_ingredients = [str(i) for i in verdict_data.get("key_ingredients", [])]
        skin_concerns = [str(c) for c in verdict_data.get("skin_concerns", [])]

        logger.info(f"⚖️ Product Verdict: {verdict} — {reason}")
        logger.info(f"🧪 Formula: {formula_texture}, color: {formula_color}")
        logger.info(f"🧴 Skin area for visuals: {skin_area}")
        logger.info(f"🧬 Key ingredients: {key_ingredients}")
        logger.info(f"💆 Skin concerns: {skin_concerns}")

        return {
            "product_research": scientific_brief,
            "product_verdict": verdict,
            "product_verdict_reason": reason,
            "formula_color": formula_color,
            "formula_texture": formula_texture,
            "skin_area": skin_area,
            "key_ingredients": key_ingredients,
            "skin_concerns": skin_concerns,
        }

    except Exception as e:
        logger.error(f"❌ Verdict Call Failed -> Error: {str(e)}")
        raise ResearchError(f"Verdict evaluation failed: {e}") from e
