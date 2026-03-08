import json
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


def _probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _merge_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Merge a Veo video clip with a TTS audio file into a single .mp4.
    Mixes Veo's native audio (SFX/ambient) with the TTS voiceover.
    If TTS is longer than the video, the last frame is frozen to cover the remainder.
    """
    audio_dur = _probe_duration(audio_path)
    video_dur = _probe_duration(video_path)
    padding = max(0.0, audio_dur - video_dur + 0.2)

    logger.debug(f"⚙️ Merge durations — video: {video_dur:.2f}s, audio: {audio_dur:.2f}s, pad: {padding:.2f}s")

    try:
        if padding > 0:
            # TTS longer than video — freeze last frame to cover the gap
            _run_ffmpeg([
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex", (
                    f"[0:v]tpad=stop_mode=clone:stop_duration={padding:.3f}[v];"
                    "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]"
                ),
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                output_path,
            ])
        else:
            # Video longer or equal — no padding needed, cut at video duration
            _run_ffmpeg([
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "0:v",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac", "-ar", "44100", "-ac", "2",
                output_path,
            ])
    except AssemblyError:
        logger.warning(f"⚠️ amix failed for {video_path} (no audio track), falling back to TTS-only audio")
        _run_ffmpeg([
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
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
