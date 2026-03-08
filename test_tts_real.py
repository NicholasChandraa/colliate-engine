import asyncio
import os
from dotenv import load_dotenv
from app.core.llm import get_genai_client
from app.graph.nodes.shot_loop import _generate_tts_audio
from app.core.config import get_settings

# Pastikan environment termuat
load_dotenv(".env")

async def test_real_tts_generation():
    print("🎙️ Testing REAL _generate_tts_audio function from shot_loop.py...")
    
    settings = get_settings()
    # TTS menggunakan client standar (tanpa project) dari llm.py
    client = get_genai_client()
    
    print(f"✅ Text-to-Speech Client Initialized.")
    
    # Siapkan folder dan path
    output_dir = os.path.join(settings.OUTPUT_DIR, "test_job_tts")
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "test_audio.wav")
    
    # Narasi bahasa Indonesia yang ingin diuji
    voiceover_text = "Rutin skincare-mu sudah benar, tapi urutan aplikasinya yang membuat kulit tetap kusam."
    
    print(f"🚀 Submitting TTS Request to Gemini ({settings.TTS_MODEL})...")
    try:
        saved_path = await _generate_tts_audio(
            client=client,
            voiceover_text=voiceover_text,
            output_path=audio_path
        )
        
        print(f"\n🎉 SUCCESS! Audio generated and saved to:\n{saved_path}")
        
    except Exception as e:
        print(f"\n❌ FAILED! Hit an error: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_tts_generation())
