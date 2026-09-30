"""Central configuration for Dimmy / DetectIQ.

Every path used by the project lives here so nothing is hardcoded to one
machine. Override any of them with an environment variable, e.g.

    set DIMMY_YOLO_WEIGHTS=D:\\models\\best.pt        (Windows)
    export DIMMY_YOLO_WEIGHTS=/models/best.pt         (Linux / macOS)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
FRONTEND_DIR = ROOT / "frontend"

# Old machine-specific location the project was originally developed in.
# Kept only as a last-resort lookup so existing trained weights are still found.
_LEGACY_ROOT = Path(r"C:\ME\Project-Detection")


def _first_existing(env_var: str, candidates: Iterable[Path]) -> Path:
    """Env var wins, otherwise the first candidate that exists, otherwise the first candidate."""
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    candidates = list(candidates)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


YOLO_WEIGHTS = _first_existing("DIMMY_YOLO_WEIGHTS", [
    ROOT / "runs" / "detect" / "idd_yolov11s_local" / "weights" / "best.pt",
    _LEGACY_ROOT / "runs" / "detect" / "idd_yolov11s_local" / "weights" / "best.pt",
])

CNN_CHECKPOINT = _first_existing("DIMMY_CNN_CHECKPOINT", [
    ROOT / "runs" / "efficientnet_train" / "best_efficientnet.pth",
    _LEGACY_ROOT / "runs" / "efficientnet_train" / "best_efficientnet.pth",
])

CNN_MAPPING = Path(os.environ.get("DIMMY_CNN_MAPPING", CNN_CHECKPOINT.parent / "class_mapping.json"))

# Datasets (only needed for training / evaluation scripts, not for the web app)
YOLO_SOURCE_DATASET = _first_existing("DIMMY_YOLO_SOURCE_DATASET", [ROOT / "IDDDetectionsYOLODataset"])
YOLO_DATASET = _first_existing("DIMMY_YOLO_DATASET", [ROOT / "IDDDetectionsYOLODataset_subset"])
CNN_RAW_DATASET = _first_existing("DIMMY_CNN_RAW_DATASET", [ROOT / "CNN_Dataset"])
CNN_DATASET = _first_existing("DIMMY_CNN_DATASET", [ROOT / "CNN_Classification_Dataset"])
RUNS_DIR = Path(os.environ.get("DIMMY_RUNS_DIR", ROOT / "runs"))

CNN_IMG_SIZE = 128

# If the trained YOLO weights are missing, fall back to the stock Ultralytics
# model so the web app still runs (it will use COCO classes, NOT the IDD ones).
# Set DIMMY_ALLOW_FALLBACK=0 to make a missing weights file a hard error instead.
ALLOW_FALLBACK_MODEL = os.environ.get("DIMMY_ALLOW_FALLBACK", "1") != "0"
FALLBACK_MODEL = os.environ.get("DIMMY_FALLBACK_MODEL", "yolo11s.pt")

YOLO_CONF = float(os.environ.get("DIMMY_CONF", "0.25"))
DEVICE = os.environ.get("DIMMY_DEVICE", "")  # "" = let Ultralytics / torch choose

# Server limits
MAX_IMAGE_MB = 25
MAX_VIDEO_MB = 200
MAX_VIDEO_FRAMES = int(os.environ.get("DIMMY_MAX_VIDEO_FRAMES", "1800"))
MAX_FRAME_WIDTH = 1280  # frames wider than this are downscaled before inference


def add_backend_paths() -> None:
    """Make the hyphen-named sub-folders importable (CNN-Part, YOLO-Part, core)."""
    for sub in ("core", "CNN-Part", "YOLO-Part"):
        p = str(BACKEND_DIR / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    b = str(BACKEND_DIR)
    if b not in sys.path:
        sys.path.insert(0, b)
