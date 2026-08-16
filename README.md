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
├── CNN_Classification_Dataset/     # 85-class cropped traffic sign dataset (train/val/test)
├── IDDDetectionsYOLODataset_subset/ # YOLO road object dataset (images & labels)
├── frontend/
│   ├── index.html                  # Futuristic dark/green HUD web interface
│   └── style.css                   # Responsive styles, glowing grid & HUD animations
├── runs/
│   ├── detect/idd_yolov11s_local/  # Trained YOLOv11 weights (best.pt)
│   └── efficientnet_train/         # Trained EfficientNet-B0 model & class_mapping.json
├── accuracy_cnn.py                 # Evaluates CNN Top-1/Top-5 accuracy & per-class F1
├── accuracy_yolo.py                # Evaluates YOLO Precision, Recall & mAP50 per class
├── detection.py                    # Main OpenCV camera loop
├── main_predict_cnn.py             # Modular EfficientNet sign classification module
├── main_predict_yolo.py            # Combined YOLO detection + CNN sign overlay logic
├── test_prediction_cnn.py          # Standalone visual test for traffic sign classifier
├── test_predict_yolo.py            # Standalone test for YOLO detection on sample images
├── prepare_CNN.py                  # Script to extract sign crops from annotated images
├── prepare_yolo.py                 # Dataset preparation and split script for YOLO
└── README.md
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DetectIQ.git
cd DetectIQ
```

### 2. Install Dependencies
```bash
pip install torch torchvision ultralytics opencv-python pillow pyyaml numpy
```

### 3. Run Live Camera Detection
```bash
python detection.py
```
*(Press `q` on the video window to safely exit).*

### 4. Evaluate Model Metrics
* **Evaluate YOLO Detection (mAP, Precision, Recall)**:
  ```bash
  python accuracy_yolo.py
  ```
* **Evaluate CNN Traffic Sign Classifier (Top-1 Accuracy, F1-Scores)**:
  ```bash
  python accuracy_cnn.py --split test
  ```

### 5. Launch the Web Interface
Open `frontend/index.html` in any web browser to view the interactive interface.

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
