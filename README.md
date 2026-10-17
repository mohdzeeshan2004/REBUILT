# Multimodal Video & Image Moderation Engine (Rebuilt)

A beginner-friendly Streamlit portfolio prototype that accepts **images and videos**, samples video frames, optionally detects common objects with YOLO, optionally transcribes audio with faster-whisper, and produces a JSON report.

## Important limitations

- This is a demo, not a production moderation service, legal opinion, or guarantee of safety.
- YOLO detects common objects; it does not reliably classify violence, nudity, hate speech, or legal compliance.
- Country profiles are placeholders for software testing and must be validated against authoritative sources before real use.
- The app intentionally never auto-approves content. If evidence is missing or an analysis module fails, the result is held for human review.

## Requirements

- Python 3.10–3.12 recommended.
- Core dependencies are listed in `requirements.txt`.
- YOLO and speech transcription are optional; their dependencies are in `requirements-optional.txt`.
- FFmpeg is optional and needed for extracting video audio.

## Install and run (Windows)

1. Extract the ZIP.
2. Open Command Prompt or PowerShell in the extracted folder (the folder containing `app.py`).
3. Run:

   ```powershell
   py -m pip install -r requirements.txt
   py -m streamlit run app.py
   ```

4. Open the local URL printed in the terminal, usually `http://localhost:8501`.
5. To enable YOLO and transcription, install optional packages separately:

   ```powershell
   py -m pip install -r requirements-optional.txt
   ```

The first YOLO/Whisper run may download model weights and can take time. If you want to test the UI first, leave the optional modules unchecked.

## Install and run (macOS/Linux)

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Optional models:

```bash
python3 -m pip install -r requirements-optional.txt
```

## Diagnose an installation issue

Run this from the project directory:

```bash
python diagnostics.py
```

If `python` is not recognized on Windows, use `py diagnostics.py`. If the app fails, copy the **full traceback** from the terminal.

## Test policy logic

```bash
python -m pip install pytest
python -m pytest -q
```

## Supported input

- Images: PNG, JPG, JPEG, WEBP, BMP
- Videos: MP4, MOV, MKV, AVI, WEBM (codec support depends on OpenCV/FFmpeg)

## Architecture

- `app.py`: Streamlit UI and workflow
- `preprocessing/video.py`: metadata, sampled frames, optional audio extraction
- `models/object_detection.py`: optional YOLO image/video object detection
- `models/speech_recognition.py`: optional faster-whisper transcription
- `moderation/evidence_fusion.py`: normalized evidence events
- `moderation/policy_engine.py`: conservative decision and review-hold logic
- `config/policies/`: demo country profiles
- `tests/`: policy-engine tests
