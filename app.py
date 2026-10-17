from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from models.object_detection import detect_image_objects, detect_objects, ModelUnavailableError
from models.speech_recognition import transcribe_audio, SpeechModelUnavailableError
from preprocessing.video import get_video_metadata, extract_frames, extract_audio
from moderation.evidence_fusion import build_events
from moderation.policy_engine import load_policy, evaluate_events

st.set_page_config(page_title="Multimodal Moderation Engine", page_icon="🎬", layout="wide")
st.title("🎬 Multimodal Video & Image Moderation Engine")
st.caption("Portfolio prototype for pre-publication screening. This is not a certified safety classifier or legal compliance system.")

with st.sidebar:
    st.header("Analysis settings")
    country = st.selectbox("Destination policy profile", ["IN", "US", "BD", "CUSTOM"], index=0)
    sample_interval = st.slider("Sample video frame every N seconds", min_value=1, max_value=10, value=3)
    run_detector = st.checkbox("Run YOLO object detection (optional model download)", value=True)
    run_transcription = st.checkbox("Transcribe video audio (optional; may download a model)", value=False)
    st.warning("Fail-closed demo: content is never auto-approved. A detected object is evidence, not proof of a policy violation. Human review is required.")

mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)

def show_result(report: dict, detections: list[dict], events: list[dict], media_type: str) -> None:
    decision = report["decision"]
    st.subheader("Moderation recommendation")
    st.warning(f"**{decision['decision'].replace('_', ' ').upper()}** — {decision['summary']}")
    left, right = st.columns(2)
    left.metric("Detections", len(detections))
    right.metric("Evidence events", len(events))
    if detections:
        st.subheader("Detected evidence")
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("No detections were returned. This does not mean the media is safe; review is still required.")
    if events:
        st.subheader("Evidence timeline")
        st.dataframe(events, use_container_width=True)
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"{media_type}_moderation_report.json",
        mime="application/json",
        key=f"download_{media_type}",
    )

if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="image_file")
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption=uploaded.name, use_container_width=True)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Could not read this image: {exc}")
            st.stop()
        if st.button("Analyze image", type="primary"):
            detections: list[dict] = []
            errors: list[str] = []
            with tempfile.TemporaryDirectory(prefix="moderation_image_") as tmp:
                image_path = Path(tmp) / "uploaded_image.png"
                image.save(image_path, format="PNG")
                if run_detector:
                    try:
                        with st.spinner("Loading object detector and analyzing image…"):
                            detections = detect_image_objects(image_path)
                    except ModelUnavailableError as exc:
                        errors.append(str(exc))
            events = [{
                "start_time": 0.0, "end_time": 0.0, "modality": "visual",
                "category": "object_detected", "label": item.get("label", "unknown"),
                "confidence": item.get("confidence"), "evidence": item,
            } for item in detections]
            if errors:
                for message in errors:
                    st.error(message)
            policy = load_policy(country)
            decision = evaluate_events(events, policy, analysis_errors=errors)
            report = {
                "media_type": "image", "filename": uploaded.name, "destination_country": country,
                "detections": detections, "events": events, "analysis_errors": errors,
                "decision": decision,
                "disclaimer": "Prototype only. Object detection is not a complete content-safety model or legal determination.",
            }
            show_result(report, detections, events, "image")
else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm"], key="video_file")
    if uploaded is not None:
        st.video(uploaded)
        st.caption(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
        if st.button("Analyze video", type="primary"):
            detections: list[dict] = []
            transcript_segments: list[dict] = []
            errors: list[str] = []
            metadata: dict = {}
            frames: list[dict] = []
            with tempfile.TemporaryDirectory(prefix="moderation_video_") as tmp:
                tmpdir = Path(tmp)
                video_path = tmpdir / f"uploaded{Path(uploaded.name).suffix.lower() or '.mp4'}"
                video_path.write_bytes(uploaded.getvalue())
                try:
                    with st.spinner("Reading video metadata and sampling frames…"):
                        metadata = get_video_metadata(video_path)
                        frames = extract_frames(video_path, tmpdir / "frames", sample_interval)
                    st.success(f"Sampled {len(frames)} frame(s).")
                except (ValueError, RuntimeError, OSError) as exc:
                    errors.append(f"Video preprocessing failed: {exc}")
                audio_path = tmpdir / "audio.wav"
                audio_ok = False
                if not errors:
                    audio_ok = extract_audio(video_path, audio_path)
                    if run_detector:
                        try:
                            with st.spinner("Running object detection on sampled frames…"):
                                detections = detect_objects(frames)
                        except ModelUnavailableError as exc:
                            errors.append(str(exc))
                    if run_transcription and audio_ok:
                        try:
                            with st.spinner("Transcribing audio…"):
                                transcript_segments = transcribe_audio(audio_path)
                        except SpeechModelUnavailableError as exc:
                            errors.append(str(exc))
                    elif run_transcription and not audio_ok:
                        errors.append("Audio transcription was requested, but FFmpeg could not extract audio. Install FFmpeg or check whether the video contains audio.")
            events = build_events(detections, transcript_segments)
            policy = load_policy(country)
            decision = evaluate_events(events, policy, analysis_errors=errors)
            report = {
                "media_type": "video", "filename": uploaded.name, "destination_country": country,
                "metadata": metadata, "sampled_frames": len(frames), "detections": detections,
                "transcript_segments": transcript_segments, "events": events, "analysis_errors": errors,
                "decision": decision,
                "disclaimer": "Prototype only. Model outputs can miss harmful content; this is not legal advice or a guarantee of safety.",
            }
            if metadata:
                with st.expander("Video metadata"):
                    st.json(metadata)
            for message in errors:
                st.error(message)
            show_result(report, detections, events, "video")

st.divider()
st.caption("Tip: first-time model use can take longer because model weights may download. For a quick dependency check, run: python diagnostics.py")
