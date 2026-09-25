from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("Install ultralytics: pip install ultralytics") from exc


class ByteTracker:

    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker_config: str = "bytetrack.yaml",
        device: str = "cpu",
        persist: bool = True,
    ) -> None:
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.iou     = iou
        self.tracker = tracker_config
        self.device  = device
        self.persist = persist

    def update(self, frame: np.ndarray) -> List[Dict]:
        results = self.model.track(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            tracker = self.tracker,
            device  = self.device,
            persist = self.persist,
            verbose = False,
        )

        tracks: List[Dict] = []
        if not results or results[0].boxes is None:
            return tracks

        boxes = results[0].boxes
        ids   = boxes.id

        for i, box in enumerate(boxes):
            track_id = int(ids[i].item()) if ids is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf    = float(box.conf[0].item())
            cls_id  = int(box.cls[0].item())
            cls_name = self.model.names.get(cls_id, str(cls_id))

            tracks.append({
                "track_id": track_id,
                "bbox":     [x1, y1, x2, y2],
                "conf":     round(conf, 3),
                "cls_id":   cls_id,
                "cls_name": cls_name,
            })

        return tracks

    def annotate(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        import cv2
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            tid   = t["track_id"]
            label = f"#{tid} {t['cls_name']} {t['conf']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return out
