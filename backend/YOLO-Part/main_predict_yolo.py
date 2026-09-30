import sys
import threading
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

config.add_backend_paths()
from main_predict_cnn import classify_sign  # noqa: E402

SIGN_KEYWORDS = ("traffic sign", "traffic light", "stop sign")

_lock = threading.Lock()
_model = None
using_fallback = False


def resolve_weights():
    """Return (weights_path, is_fallback) for the model that should be loaded."""
    weights = Path(config.YOLO_WEIGHTS)
    if weights.exists():
        return str(weights), False
    if config.ALLOW_FALLBACK_MODEL:
        print(f"[!] Trained weights not found at {weights}. "
              f"Falling back to stock '{config.FALLBACK_MODEL}' (COCO classes).")
        return config.FALLBACK_MODEL, True
    raise FileNotFoundError(f"YOLO weights not found: {weights}")


def get_model():
    """Load the YOLO model once, on first use."""
    global _model, using_fallback
    with _lock:
        if _model is None:
            path, using_fallback = resolve_weights()
            _model = YOLO(path)
        return _model


def predict_frame(frame, return_detections=False):
    """Run YOLO (+ CNN sign classification) on one BGR frame.

    Returns the annotated frame, or (annotated_frame, detections) when
    return_detections=True.
    """
    model = get_model()
    kwargs = {"conf": config.YOLO_CONF, "verbose": False}
    if config.DEVICE:
        kwargs["device"] = config.DEVICE
    results = model(frame, **kwargs)
    annotated_frame = results[0].plot()

    detections = []
    h, w = frame.shape[:2]

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det = {
            "label": label,
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [x1, y1, x2, y2],
            "sign": None,
        }

        if any(k in label.lower() for k in SIGN_KEYWORDS):
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            sign_crop = frame[cy1:cy2, cx1:cx2]
            sign_name, conf = classify_sign(sign_crop)

            if sign_name != "UNKNOWN":
                det["sign"] = {"name": sign_name, "confidence": round(conf, 1)}
                cv.putText(
                    annotated_frame,
                    f"{sign_name} ({conf:.0f}%)",
                    (cx1, max(20, cy1 - 10)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

        detections.append(det)

    if return_detections:
        return annotated_frame, detections
    return annotated_frame
