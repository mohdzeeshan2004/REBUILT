from __future__ import annotations


def build_events(detections: list[dict], transcript_segments: list[dict]) -> list[dict]:
    """Normalize model outputs. These are evidence records, not violation findings."""
    events: list[dict] = []
    for item in detections:
        events.append({
            "start_time": float(item.get("timestamp", 0) or 0),
            "end_time": float(item.get("timestamp", 0) or 0),
            "modality": "visual", "category": "object_detected",
            "label": str(item.get("label", "unknown")).lower(),
            "confidence": item.get("confidence"), "evidence": item,
        })
    for item in transcript_segments:
        events.append({
            "start_time": float(item.get("start_time", 0) or 0),
            "end_time": float(item.get("end_time", 0) or 0),
            "modality": "audio_text", "category": "transcript_available",
            "label": str(item.get("text", "")), "confidence": None, "evidence": item,
        })
    return sorted(events, key=lambda event: event["start_time"])
