import os
import subprocess
from app.core.exceptions import AssemblyError
from app.core.logging import get_logger
from app.graph.state import GraphState

logger = get_logger(__name__)


def _run_ffmpeg(args: list[str]) -> None:
    """Run an FFmpeg command. Raises AssemblyError on non-zero exit."""
    cmd = ["ffmpeg", "-y", *args]
    logger.debug(f"⚙️ FFmpeg Engine Subprocess -> Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssemblyError(f"FFmpeg failed:\n{result.stderr}")


def _concatenate_clips(clip_paths: list[str], output_path: str) -> None:
    """Concatenate multiple .mp4 clips into one final video."""
    concat_list_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")

    with open(concat_list_path, "w") as f:
        for clip in clip_paths:
            _ = f.write(f"file '{os.path.abspath(clip)}'\n")

    _run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",           # no re-encode, just stitch
        output_path,
    ])


def _merge_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Merge a Veo video clip with a TTS audio file into a single .mp4.
    Mixes Veo's native audio (SFX/ambient) with the TTS voiceover.
    """
    try:
        # Try to mix both audio streams (assuming Veo generated an audio track)
        _run_ffmpeg([
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            output_path,
        ])
    except AssemblyError:
        logger.warning(f"⚠️ amix failed for {video_path}, falling back to replacing audio track")
        # Fallback if video has no audio track: just use the TTS audio
        _run_ffmpeg([
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-shortest",
            output_path,
        ])
    logger.debug(f"⚙️ FFmpeg Merge Complete -> {output_path}")
    return output_path


def assembly_node(state: GraphState) -> dict[str, str]:
    logger.info(f"🎬 Graph Node: Final Assembly Started -> Merging {len(state.generated_video_paths)} clips")

    # Veo 3.1 already produces video with native audio — just concatenate
    output_dir = os.path.dirname(state.generated_video_paths[0])
    final_video_path = os.path.join(output_dir, "final_video.mp4")

    _concatenate_clips(state.generated_video_paths, final_video_path)

    logger.info(f"✅ Final Assembly Completed -> Master Output: {final_video_path}")

    return {"final_video_path": final_video_path}
