from __future__ import annotations
from pathlib import Path
from typing import Any

class ModelUnavailableError(RuntimeError):
    """Raised when the optional YOLO dependency or weights cannot be loaded."""


def _load_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelUnavailableError(
            "YOLO is not installed. Install optional AI dependencies with: python -m pip install -r requirements-optional.txt. "
            "The app will still create a review-hold report without the model."
        ) from exc
    try:
        return YOLO(model_name)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Could not load YOLO model '{model_name}'. Check internet access for first-time weight download, or provide a local model file. Details: {exc}"
        ) from exc


def _parse_result(result: Any, timestamp: float | None = None, model_name: str = "yolov8n.pt") -> list[dict]:
    output: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return output
    names = getattr(result, "names", {})
    for box in boxes:
        try:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coords = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            item = {"label": str(names.get(class_id, class_id)), "confidence": round(confidence, 4), "bbox_xyxy": coords, "model": model_name}
            if timestamp is not None:
                item["timestamp"] = round(float(timestamp), 3)
            output.append(item)
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
    return output


def detect_image_objects(image_path: str | Path, model_name: str = "yolov8n.pt") -> list[dict]:
    model = _load_model(model_name)
    try:
        result = model.predict(source=str(image_path), verbose=False)[0]
    except Exception as exc:
        raise ModelUnavailableError(f"YOLO failed while analyzing the image: {exc}") from exc
    return _parse_result(result, model_name=model_name)


def detect_objects(frames: list[dict], model_name: str = "yolov8n.pt") -> list[dict]:
    if not frames:
        return []
    model = _load_model(model_name)
    output: list[dict] = []
    for frame in frames:
        try:
            result = model.predict(source=str(frame["path"]), verbose=False)[0]
        except Exception as exc:
            raise ModelUnavailableError(f"YOLO failed on frame at {frame.get('timestamp', 0)} seconds: {exc}") from exc
        output.extend(_parse_result(result, timestamp=float(frame.get("timestamp", 0)), model_name=model_name))
    return output
