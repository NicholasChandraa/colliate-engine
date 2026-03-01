from typing import cast
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import get_settings
from app.core.exceptions import ResearchError
from app.core.logging import logger
from app.graph.state import GraphState

def _build_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the LangChain Google Generative AI model
    used for the Research node, configured to run with a deterministically low temperature (0.2).
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.LLM_RESEARCH_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2
    )

RESEARCH_PROMPT = """
Kamu adalah seorang senior product research kreatif yang spesialis dalam pembuatan iklan video digital.

Product : {product_name}
Target : {target_audience}

Gunakan Google Search untuk mencari informasi terkini dan akurat tentang produk ini sebelum memberikan analisis.
Pastikan data yang kamu berikan berdasarkan hasil pencarian web, bukan hanya pengetahuan internal kamu.

Lakukan riset mendalam dan kembalikan output dalam format berikut (gunakan bahasa Indonesia):

## KEY FEATURES & BENEFITS
Fitur atau kandungan utama produk beserta manfaat langsungnya bagi konsumen.

## UNIQUE SELLING POINTS (top 3)
Kelebihan (USP) spesifik yang membedakan produk ini dari kompetitor di pasaran.

## PAIN POINTS TARGET AUDIENCE
Permasalahan utama, kekhawatiran, atau kebutuhan spesifik yang paling sering dirasakan oleh target audience ini.

## VIDEO HOOK IDEAS (top 3)
Kalimat pembuka yang kuat untuk menarik perhatian dalam 3 detik pertama.

## RECOMMENDED VIDEO TONE
Pilih satu: Fun / Emotional / Educational / Aspirational - beserta alasannya.
"""

async def research_node(state: GraphState) -> dict[str, str]:
    """
    LangGraph Node: Researches and builds marketing context for the commercial video.
    
    This node analyzes the input 'product_name' and 'target_audience' to define the 
    core problems, desired solutions, and the overarching marketing angle for the commercial.
    
    Args:
        state: The current initialized state of the graph containing the user's base inputs.
        
    Returns:
        A dictionary containing the 'product_research' string output which serves
        as a guiding context for the downstream 'director' node.
        
    Raises:
        ResearchError: If the LLM encounters an issue or the response is completely null.
    """
    logger.info(f"🎬 Graph Node: Web Researcher Started -> Crawling info for '{state.product_name}'")

    llm = _build_llm().bind_tools([{"google_search": {}}])
    prompt = RESEARCH_PROMPT.format(
        product_name=state.product_name,
        target_audience=state.target_audience,
    )

    try:
        logger.info(f"📑 LLM PROMPT RESEARCHER: {prompt}")
        llm_response = await llm.ainvoke(prompt)
        logger.info(f"📑 LLM RESEARCHER RESPONSE: {llm_response}")

        # Log grounding metadata to confirm Google Search was used
        grounding = llm_response.response_metadata.get("grounding_metadata") if llm_response.response_metadata else None
        if grounding:
            search_queries = grounding.get("web_search_queries", [])
            logger.info(f"🔍 Google Search Used -> Queries: {search_queries}")
        else:
            logger.warning("⚠️ Google Search NOT used — response has no grounding_metadata")

        raw = llm_response.content
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            content = cast(str, raw[0].get("text", ""))
        else:
            content = str(raw)

        logger.info(f"✅ Web Researcher Completed -> Gathered {len(content)} characters of intel.")
        return {"product_research": content}
        
    except Exception as e:
        logger.error(f"❌ Web Researcher Failed -> Error: {str(e)}")
        raise ResearchError(f"Research failed: {e}") from e