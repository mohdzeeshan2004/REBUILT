from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
import cv2


def get_video_metadata(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video. Check that the file is valid and supported codecs are installed.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("The video has no readable frames or has unsupported encoding.")
    duration = frame_count / fps if fps > 0 else None
    return {"fps": round(fps, 3), "frame_count": frame_count, "width": width, "height": height, "duration_seconds": round(duration, 3) if duration is not None else None}


def extract_frames(video_path: str | Path, output_dir: str | Path, interval_seconds: int = 3) -> list[dict]:
    if interval_seconds < 1:
        raise ValueError("Frame interval must be at least one second.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video for frame extraction.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        cap.release()
        raise ValueError("Video FPS is unavailable; cannot sample frames.")
    stride = max(1, int(round(fps * interval_seconds)))
    frames: list[dict] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                timestamp = index / fps
                path = output_dir / f"frame_{timestamp:09.3f}.jpg"
                if cv2.imwrite(str(path), frame):
                    frames.append({"timestamp": round(timestamp, 3), "path": str(path)})
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No frames could be extracted from this video.")
    return frames


def extract_audio(video_path: str | Path, output_path: str | Path) -> bool:
    """Extract mono 16kHz WAV with FFmpeg. False means unavailable/no audio."""
    if shutil.which("ffmpeg") is None:
        return False
    command = ["ffmpeg", "-nostdin", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(output_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 44
    except (OSError, subprocess.TimeoutExpired):
        return False
