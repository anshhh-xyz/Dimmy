import os
import sys
import json
import threading
from pathlib import Path

import cv2 as cv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CNN_CHECKPOINT = str(config.CNN_CHECKPOINT)
MAPPING_PATH = str(config.CNN_MAPPING)
IMG_SIZE = config.CNN_IMG_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# The model is loaded lazily on first use. Previously it was loaded at import
# time, which meant merely importing this file crashed (or silently produced a
# randomly-initialised network) when the checkpoint was missing.
_lock = threading.Lock()
_model = None
_class_names = []
_load_error = None


def _load():
    global _model, _class_names, _load_error
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            if not os.path.exists(CNN_CHECKPOINT):
                raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CHECKPOINT}")

            checkpoint = torch.load(CNN_CHECKPOINT, map_location=device)

            names = []
            if os.path.exists(MAPPING_PATH):
                with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                if "idx_to_class" in mapping:
                    idx_map = mapping["idx_to_class"]
                    names = [idx_map[str(i)] for i in range(len(idx_map))]
            if not names and isinstance(checkpoint, dict) and "class_names" in checkpoint:
                names = list(checkpoint["class_names"])

            state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            if not names:
                # Infer the number of classes from the saved classifier layer
                num_classes = state["classifier.1.weight"].shape[0]
                names = [f"Class_{i}" for i in range(num_classes)]

            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(names))
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()

            _class_names = names
            _model = model
        except Exception as exc:  # noqa: BLE001
            _load_error = str(exc)


def is_available() -> bool:
    _load()
    return _model is not None


def load_error():
    _load()
    return _load_error


def classify_sign(crop_bgr):
    """Classify a cropped traffic sign. Returns (class_name, confidence_percent)."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return "UNKNOWN", 0.0

    _load()
    if _model is None:
        return "UNKNOWN", 0.0

    rgb_img = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = _model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.topk(probs, 1)

    cls_idx = top_idx[0].item()
    conf = top_prob[0].item() * 100
    cls_name = _class_names[cls_idx] if cls_idx < len(_class_names) else f"Class_{cls_idx}"
    return cls_name, conf
