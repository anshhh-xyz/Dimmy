"""Dimmy / DetectIQ web API.

Run from the project root:

    pip install -r requirements.txt
    python backend/server.py

then open http://localhost:8000  (the server also serves the frontend).

Endpoints
    GET  /api/health          status + which model files were found
    POST /api/detect/image    multipart 'file'            -> annotated image + detections
    POST /api/detect/video    multipart 'file'            -> annotated video URL + risk timeline
    GET  /api/video/{id}      the processed video
    POST /api/live/frame      multipart 'file', form 'session' -> annotated frame + live risk
    POST /api/live/reset      form 'session'              -> clears tracking state
"""
from __future__ import annotations

import base64
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

config.add_backend_paths()

app = FastAPI(title="Dimmy API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lets frontend/index.html work when opened straight from disk too
    allow_methods=["*"],
    allow_headers=["*"],
)

# One inference at a time: YOLO's tracker state and the shared model are not thread-safe.
_infer_lock = threading.Lock()

VIDEO_DIR = Path(tempfile.gettempdir()) / "dimmy_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_TTL_S = 3600


# ────────────────────────── helpers ──────────────────────────

def _clean(obj: Any) -> Any:
    """Make results JSON-safe: inf/NaN -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


def _risk_level(pct: float) -> str:
    if pct >= 70:
        return "CRITICAL"
    if pct >= 20:
        return "CAUTION"
    return "SAFE"


def _read_upload(file: UploadFile, max_mb: int) -> bytes:
    data = file.file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {max_mb} MB)")
    if not data:
        raise HTTPException(400, "Empty file")
    return data


def _decode_image(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image. Use JPG, PNG or WEBP.")
    return frame


def _limit_width(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= config.MAX_FRAME_WIDTH:
        return frame
    scale = config.MAX_FRAME_WIDTH / w
    return cv2.resize(frame, (config.MAX_FRAME_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def _to_data_url(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "Could not encode result image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _device() -> str:
    if config.DEVICE:
        return config.DEVICE
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _yolo():
    """Import the model code lazily so /api/health works even if torch is missing."""
    try:
        import main_predict_yolo
        return main_predict_yolo
    except ImportError as exc:
        raise HTTPException(503, f"Model dependencies missing ({exc}). Run: pip install -r requirements.txt")


def _new_pipeline():
    from adas_pipeline import ADASPipeline
    path, _ = _yolo().resolve_weights()
    return ADASPipeline(model_path=path, device=_device())


def _summarize_adas(results: Dict) -> Dict:
    """Flatten ADAS output into something the UI can show."""
    objects = []
    track_by_id = {t["track_id"]: t for t in results["tracks"]}
    for tid, risk in results["risk"].items():
        t = track_by_id.get(tid, {})
        sp = results["speeds"].get(tid, {})
        objects.append({
            "track_id": tid,
            "label": t.get("cls_name"),
            "bbox": t.get("bbox"),
            "speed_kmh": sp.get("speed_kmh"),
            "distance_m": risk["distance_m"],
            "ttc_s": risk["ttc_s"],
            "risk_pct": risk["risk_pct"],
            "risk_level": risk["risk_level"],
        })
    objects.sort(key=lambda o: o["risk_pct"], reverse=True)
    max_risk = float(results["max_risk_pct"])
    return {
        "max_risk_pct": max_risk,
        "risk_level": _risk_level(max_risk),
        "object_count": len(results["tracks"]),
        "objects": objects[:10],
        "signs": [{"name": s["name"], "confidence": s["confidence"]} for s in results["signs"]],
        "class_counts": dict(Counter(t["cls_name"] for t in results["tracks"])),
    }


# ────────────────────────── routes ──────────────────────────

@app.get("/api/health")
def health():
    weights = Path(config.YOLO_WEIGHTS)
    cnn = Path(config.CNN_CHECKPOINT)
    return {
        "status": "ok",
        "yolo_weights": str(weights),
        "yolo_weights_found": weights.exists(),
        "using_fallback_model": (not weights.exists()) and config.ALLOW_FALLBACK_MODEL,
        "cnn_checkpoint": str(cnn),
        "cnn_checkpoint_found": cnn.exists(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.post("/api/detect/image")
def detect_image(file: UploadFile = File(...)):
    frame = _limit_width(_decode_image(_read_upload(file, config.MAX_IMAGE_MB)))
    mod = _yolo()
    t0 = time.perf_counter()
    try:
        with _infer_lock:
            annotated, detections = mod.predict_frame(frame, return_detections=True)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    ms = (time.perf_counter() - t0) * 1000

    return _clean({
        "image": _to_data_url(annotated),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "detections": detections,
        "class_counts": dict(Counter(d["label"] for d in detections)),
        "signs": [d["sign"] | {"bbox": d["bbox"]} for d in detections if d["sign"]],
        "inference_ms": round(ms, 1),
        "using_fallback_model": mod.using_fallback,
    })


@app.post("/api/detect/video")
def detect_video(file: UploadFile = File(...)):
    data = _read_upload(file, config.MAX_VIDEO_MB)
    _purge_old_videos()

    job = uuid.uuid4().hex
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    src = VIDEO_DIR / f"{job}_in{suffix}"
    raw = VIDEO_DIR / f"{job}_raw.mp4"
    out = VIDEO_DIR / f"{job}.mp4"
    src.write_bytes(data)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        src.unlink(missing_ok=True)
        raise HTTPException(400, "Could not open video. Try MP4, AVI, MOV or MKV.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps < 1:
        fps = 30.0

    writer = None
    timeline = []
    class_counts: Counter = Counter()
    signs_seen: Dict[str, float] = {}
    peak = {"risk": 0.0, "t": 0.0}
    frames = 0
    truncated = False

    try:
        with _infer_lock:
            pipeline = _new_pipeline()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frames >= config.MAX_VIDEO_FRAMES:
                    truncated = True
                    break

                frame = _limit_width(frame)
                t = frames / fps
                annotated, res = pipeline.process(frame, timestamp=t)

                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise HTTPException(500, "Could not create output video")
                writer.write(annotated)

                risk = float(res["max_risk_pct"])
                timeline.append([round(t, 2), risk])
                if risk > peak["risk"]:
                    peak = {"risk": risk, "t": round(t, 2)}
                for tr in res["tracks"]:
                    class_counts[tr["cls_name"]] += 1
                for s in res["signs"]:
                    signs_seen[s["name"]] = max(signs_seen.get(s["name"], 0.0), s["confidence"])
                frames += 1
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        src.unlink(missing_ok=True)

    if frames == 0:
        raw.unlink(missing_ok=True)
        raise HTTPException(400, "No frames could be read from this video")

    browser_ok = _transcode_h264(raw, out)
    if not browser_ok:
        shutil.move(str(raw), str(out))

    # keep the timeline small enough to send over the wire
    step = max(1, len(timeline) // 120)
    return _clean({
        "video_url": f"/api/video/{job}",
        "browser_playable": browser_ok,
        "frames": frames,
        "fps": round(fps, 2),
        "duration_s": round(frames / fps, 2),
        "truncated": truncated,
        "max_risk_pct": peak["risk"],
        "risk_level": _risk_level(peak["risk"]),
        "peak_risk_time_s": peak["t"],
        "risk_timeline": timeline[::step],
        "class_counts": dict(class_counts.most_common(12)),
        "signs": [{"name": n, "confidence": c} for n, c in sorted(signs_seen.items(), key=lambda kv: -kv[1])],
    })


@app.get("/api/video/{video_id}")
def get_video(video_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", video_id):
        raise HTTPException(400, "Bad video id")
    path = VIDEO_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Video expired or not found")
    return FileResponse(path, media_type="video/mp4", filename="dimmy_result.mp4")


# ── live camera ──
_sessions: Dict[str, Dict] = {}
_MAX_SESSIONS = 4
_SESSION_TTL_S = 600


def _get_session(sid: str):
    now = time.time()
    for k in [k for k, v in _sessions.items() if now - v["used"] > _SESSION_TTL_S]:
        del _sessions[k]
    if sid not in _sessions:
        while len(_sessions) >= _MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k]["used"])
            del _sessions[oldest]
        _sessions[sid] = {"pipeline": _new_pipeline(), "used": now}
    _sessions[sid]["used"] = now
    return _sessions[sid]["pipeline"]


@app.post("/api/live/frame")
def live_frame(file: UploadFile = File(...), session: str = Form("default")):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session):
        raise HTTPException(400, "Bad session id")
    frame = _limit_width(_decode_image(_read_upload(file, config.MAX_IMAGE_MB)))
    t0 = time.perf_counter()
    try:
        with _infer_lock:
            pipeline = _get_session(session)
            annotated, res = pipeline.process(frame)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    ms = (time.perf_counter() - t0) * 1000

    summary = _summarize_adas(res)
    summary["image"] = _to_data_url(annotated, quality=70)
    summary["inference_ms"] = round(ms, 1)
    return _clean(summary)


@app.post("/api/live/reset")
def live_reset(session: str = Form("default")):
    with _infer_lock:
        s = _sessions.get(session)
        if s:
            s["pipeline"].reset()
    return {"status": "reset"}


# ── video helpers ──

def _transcode_h264(src: Path, dst: Path) -> bool:
    """OpenCV's mp4v codec won't play in browsers; re-encode to H.264 if ffmpeg exists."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-movflags", "+faststart", "-an", str(dst)],
            capture_output=True, timeout=600,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0 or not dst.exists():
        return False
    src.unlink(missing_ok=True)
    return True


def _purge_old_videos() -> None:
    now = time.time()
    for p in VIDEO_DIR.glob("*"):
        try:
            if now - p.stat().st_mtime > VIDEO_TTL_S:
                p.unlink()
        except OSError:
            pass


# ── frontend (mounted last so /api/* wins) ──
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("DIMMY_HOST", "127.0.0.1")
    port = int(os.environ.get("DIMMY_PORT", "8000"))
    print(f"Dimmy running on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
