from __future__ import annotations
from pathlib import Path

class SpeechModelUnavailableError(RuntimeError):
    """Raised when optional speech transcription cannot run."""


def transcribe_audio(audio_path: str | Path, model_size: str = "base") -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SpeechModelUnavailableError(
            "faster-whisper is not installed. Install optional AI dependencies with: python -m pip install -r requirements-optional.txt."
        ) from exc
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio_path), vad_filter=True)
        output = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                output.append({"start_time": round(float(segment.start), 3), "end_time": round(float(segment.end), 3), "text": text, "language": info.language, "model": f"faster-whisper-{model_size}"})
        return output
    except Exception as exc:
        raise SpeechModelUnavailableError(f"Speech transcription failed: {exc}") from exc
