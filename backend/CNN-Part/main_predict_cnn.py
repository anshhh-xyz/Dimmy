import os
import json
import cv2 as cv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

CNN_CHECKPOINT = r"C:\ME\Project-Detection\runs\efficientnet_train\best_efficientnet.pth"
MAPPING_PATH = r"C:\ME\Project-Detection\runs\efficientnet_train\class_mapping.json"
IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = []
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        if "idx_to_class" in mapping:
            idx_map = mapping["idx_to_class"]
            class_names = [idx_map[str(i)] for i in range(len(idx_map))]

if not class_names and os.path.exists(CNN_CHECKPOINT):
    checkpoint_data = torch.load(CNN_CHECKPOINT, map_location="cpu")
    if "class_names" in checkpoint_data:
        class_names = checkpoint_data["class_names"]

num_classes = len(class_names) if class_names else 85

cnn_model = models.efficientnet_b0(weights=None)
in_features = cnn_model.classifier[1].in_features
cnn_model.classifier[1] = nn.Linear(in_features, num_classes)

if os.path.exists(CNN_CHECKPOINT):
    checkpoint = torch.load(CNN_CHECKPOINT, map_location=device)
    if "model_state_dict" in checkpoint:
        cnn_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        cnn_model.load_state_dict(checkpoint)

cnn_model = cnn_model.to(device)
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_sign(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
        return "UNKNOWN", 0.0

    rgb_img = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_prob, top_idx = torch.topk(probs, 1)

    cls_idx = top_idx[0].item()
    conf = top_prob[0].item() * 100
    cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"

    return cls_name, conf
