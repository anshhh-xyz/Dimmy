# 🚗 DetectIQ — Intelligent Road Object Detection & ADAS Vision Stack

> **A Hybrid Two-Stage Computer Vision Pipeline with Real-Time Traffic Sign Classification and Upcoming Autonomous Collision Risk Estimation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11s-00ffff.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![EfficientNet](https://img.shields.io/badge/Model-EfficientNet--B0-brightgreen.svg)](https://pytorch.org/vision/stable/models/efficientnet.html)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8.svg?logo=opencv&logoColor=white)](https://opencv.org/)

---

## 📌 Project Overview

**DetectIQ** is a computer vision and deep learning system engineered specifically for unstructured, high-density traffic environments (such as Indian roadways). The system combines state-of-the-art **YOLOv11s** object localization with an **EfficientNet-B0** deep classifier to detect vehicles, vulnerable road users, and accurately identify 85+ distinct road signs in real time.

We are currently expanding DetectIQ into a full **Advanced Driver Assistance System (ADAS)** by introducing multi-object tracking, monocular speed calculation, and predictive **Time-to-Collision (TTC)** accident risk estimation.

---

## 🏗️ Current Architecture (Two-Stage Pipeline)

```
                       [ Live Video / Camera Frame ]
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    Stage 1: YOLOv11s Model   │
                      │  (Localizes vehicles, signs, │
                      │    pedestrians & obstacles)  │
                      └──────────────┬───────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
       [ Standard Objects ]                      [ Traffic Sign / Light ]
       (Cars, Bikes, Buses)                              │
                 │                                       ▼
                 │                       ┌──────────────────────────────┐
                 │                       │  Dynamic Bounding Box Crop   │
                 │                       └──────────────┬───────────────┘
                 │                                      │
                 │                                      ▼
                 │                       ┌──────────────────────────────┐
                 │                       │ Stage 2: EfficientNet-B0 CNN │
                 │                       │ (85-Class Fine-Grained Sign  │
                 │                       │        Classification)       │
                 │                       └──────────────┬───────────────┘
                 │                                      │
                 └───────────────────┬──────────────────┘
                                     │
                                     ▼
                    [ Fully Annotated Real-Time Output ]
```

### 1. Stage 1: Spatial Object Localization (YOLOv11s)
* Trained on the **Indian Driving Dataset (IDD)**.
* Detects vehicles, riders, pedestrians, and road infrastructure in complex, dense traffic scenes.
* Outputs bounding box coordinates `[x1, y1, x2, y2]`, object class IDs, and confidence scores at **~30 FPS**.

### 2. Stage 2: Fine-Grained Traffic Sign Classification (EfficientNet-B0)
* Trained on a dedicated dataset of **85 distinct Indian traffic signs and signals** (`CNN_Classification_Dataset`).
* When YOLO spots a `traffic sign` or `traffic light`, the bounding box region is dynamically cropped and passed to the CNN.
* Output: Identifies the exact sign name (e.g., `STOP`, `SPEED_LIMIT_60`, `NO_ENTRY`, `PEDESTRIAN_CROSSING`, `TRAFFIC_SIGNAL`) with confidence percentages.

### 3. Cyberpunk / HUD Web Dashboard
* Pure black & neon green futuristic interface (`frontend/index.html` & `frontend/style.css`).
* Supports **Image Upload**, **Video Processing**, and **Live Webcam Feed**.
* Integrated telemetry display and accident risk indicator gauge.

---

## 🚀 What We Are Adding Next (ADAS Roadmap)

The next phase transitions DetectIQ from passive detection into an active **Autonomous Driving & Safety Assistance Stack**:

```
[ YOLO Detection ] ──► [ ByteTrack Multi-Object Tracking ] ──► [ Inverse Perspective Mapping (IPM) ]
                                                                             │
                                                                             ▼
[ 0-100% Risk Gauge ] ◄── [ Time-to-Collision (TTC) Engine ] ◄── [ Real-World Velocity (km/h) ]
```

### 1. Multi-Object Tracking (MOT) with ByteTrack
* **Goal**: Assign persistent `track_id`s to vehicles across consecutive frames.
* **Method**: Leverage Kalman filters and bipartite matching (IoU + appearance features) via Ultralytics ByteTrack integration (`model.track`).

### 2. Monocular Speed & Distance Estimation
* **Goal**: Estimate approaching and departing vehicle speeds (in km/h) using a single dashboard camera without expensive LiDAR.
* **Method**:
  * **Inverse Perspective Mapping (IPM / Homography)** using `cv2.getPerspectiveTransform()` to project distorted 2D camera pixels into a top-down ground coordinate plane in meters.
  * Compute velocity vector: $v = \frac{\Delta \text{distance (meters)}}{\Delta t \text{ (seconds)}} \times 3.6\text{ km/h}$.

### 3. Predictive Accident & Collision Probability Engine
* **Goal**: Dynamically assess and warn drivers of imminent collision risk.
* **Method**:
  * **Time-to-Collision (TTC)** formulation:
    $$\text{TTC} = \frac{\text{Distance to Target (m)}}{\text{Relative Approach Speed (m/s)}}$$
  * **Risk Scoring Model**:
    * **TTC > 5.0s** $\rightarrow$ `Safe (0 - 20% Risk)`
    * **2.5s < TTC ≤ 5.0s** $\rightarrow$ `Moderate / Caution (20 - 70% Risk)`
    * **TTC ≤ 2.5s** $\rightarrow$ `CRITICAL COLLISION WARNING (70 - 100% Risk)`

### 4. Full Dataset YOLO Retraining (30,000 Images)
* Retraining YOLOv11s on the complete 30k IDD image split.
* Pruning redundant/rare classes to concentrate model capacity on: `car`, `motorcycle`, `bus`, `truck`, `autorickshaw`, `person`, `rider`, `traffic sign`, and `traffic light`.
* Oversampling sign-containing scenes to significantly boost recall on small traffic signals.

---

## 📂 Repository Structure

```
├── backend/
│   ├── config.py                   # All paths/settings (env-overridable, no hardcoded C:\ paths)
│   ├── server.py                   # FastAPI server: REST API + serves the frontend
│   ├── CNN-Part/                   # EfficientNet-B0 sign classifier (prepare / train / predict / evaluate)
│   ├── YOLO-Part/                  # YOLOv11s detector (prepare / train / predict / evaluate)
│   └── core/                       # ADAS stack: tracker, speed (IPM), TTC, risk, pipeline, webcam demo
├── frontend/
│   ├── index.html                  # Web UI (image / video / live camera), calls the API
│   └── style.css
├── runs/                           # (not in git) detect/idd_yolov11s_local/weights/best.pt
│                                   #               efficientnet_train/best_efficientnet.pth + class_mapping.json
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Put your trained weights in place
Copy your `runs/` folder into the project root so these exist:
```
runs/detect/idd_yolov11s_local/weights/best.pt
runs/efficientnet_train/best_efficientnet.pth
runs/efficientnet_train/class_mapping.json
```
or point to them with environment variables (`DIMMY_YOLO_WEIGHTS`, `DIMMY_CNN_CHECKPOINT`, `DIMMY_CNN_MAPPING`).
If the YOLO weights are missing the app falls back to the stock `yolo11s.pt` (generic COCO classes) and the
navbar shows **"Online · stock model"**. Set `DIMMY_ALLOW_FALLBACK=0` to disable that.

### 3. Run the web app
```bash
python backend/server.py
```
Open **http://localhost:8000**.  Image upload, video processing and live camera all run through the API.

### 4. API
| Method | Path | Purpose |
| :--- | :--- | :--- |
| GET | `/api/health` | Status + which model files were found |
| POST | `/api/detect/image` | `file` → annotated image, detections, classified signs |
| POST | `/api/detect/video` | `file` → processed video URL, peak risk, risk timeline |
| GET | `/api/video/{id}` | Download / stream processed video |
| POST | `/api/live/frame` | `file`, `session` → annotated frame + live accident risk |
| POST | `/api/live/reset` | `session` → clear tracking state |

Interactive docs: http://localhost:8000/docs

### 5. Other scripts (run from the project root)
```bash
python backend/core/detection.py                 # OpenCV webcam window, press q to quit
python backend/YOLO-Part/accuracy_yolo.py        # YOLO precision / recall / mAP
python backend/CNN-Part/accuracy_cnn.py --split test
```
Optional: install `ffmpeg` so processed videos preview in the browser (otherwise they are download-only).

---

## 📊 Performance Summary

| Model Component | Architecture | Dataset | Metric |
| :--- | :--- | :--- | :--- |
| **Object Detector** | YOLOv11s | IDD Subset | Real-Time Inference (~30 FPS) |
| **Sign Classifier** | EfficientNet-B0 | 85 Indian Sign Classes | Top-1 Accuracy / 85 Fine Classes |
| **ADAS Vision Engine** | ByteTrack + IPM | Real-World Kinematics | *In Development (Phase 2)* |

---

## 👥 Authors & Acknowledgments

* **Developer**: Built as an intelligent transportation & ADAS vision research project.
* **Datasets**: Indian Driving Dataset (IDD), Traffic Sign Benchmarks.
* **Frameworks**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/).
