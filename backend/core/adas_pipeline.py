from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

config.add_backend_paths()

from tracker               import ByteTracker                                   # noqa: E402
from speed_detection       import SpeedEstimator                                # noqa: E402
from time_collide          import TTCEngine                                     # noqa: E402
from probability_calculator import AccidentProbabilityCalculator, draw_risk_overlay  # noqa: E402

# Only things a car can actually collide with take part in speed/TTC/risk.
# Signs and traffic lights are stationary, so as the ego vehicle approaches them
# they looked like "objects closing in fast" and produced false CRITICAL alerts.
HAZARD_CLASSES = {
    "animal", "autorickshaw", "bicycle", "bus", "car", "caravan", "motorcycle",
    "person", "rider", "trailer", "train", "truck", "vehicle fallback",
    # COCO names, used when the stock fallback model is loaded
    "motorbike", "bike", "cow", "dog", "horse", "sheep",
}
SIGN_CLASSES = ("traffic sign", "traffic light", "stop sign")


class ADASPipeline:

    def __init__(
        self,
        model_path:       str,
        ipm_src_pts:      Optional[List] = None,
        ipm_dst_size_m:   Tuple[float, float] = (6.0, 20.0),
        ego_speed_kmh:    float = 0.0,
        smoothing_window: int   = 5,
        device:           str   = "cpu",
        model=None,
        classify_signs:   bool  = True,
    ) -> None:
        self.tracker   = ByteTracker(model_path, device=device, model=model)
        self.speed_est = SpeedEstimator(
            src_pts          = ipm_src_pts,
            dst_size_m       = ipm_dst_size_m,
            smoothing_window = smoothing_window,
        )
        self.ttc_engine    = TTCEngine()
        self.risk_calc     = AccidentProbabilityCalculator()
        self.ego_speed_kmh = ego_speed_kmh
        self.classify_signs = classify_signs

    def reset(self) -> None:
        self.tracker.reset()
        self.speed_est.reset()

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        h, w = frame.shape[:2]
        self.speed_est.set_frame_size(w, h)

        tracks = self.tracker.update(frame)

        hazards = [
            t for t in tracks
            if t["cls_name"].lower() in HAZARD_CLASSES and t["track_id"] >= 0
        ]

        speed_map = self.speed_est.update(hazards, timestamp=timestamp)
        ttc_map   = self.ttc_engine.update(speed_map, ego_speed_kmh=self.ego_speed_kmh)
        risk_map  = self.risk_calc.compute(ttc_map)

        bbox_map: Dict[int, list] = {int(t["track_id"]): t["bbox"] for t in hazards}

        signs = self._classify_signs(frame, tracks) if self.classify_signs else []

        annotated = self.tracker.annotate(frame, tracks)
        annotated = draw_risk_overlay(annotated, bbox_map, risk_map)
        annotated = self._draw_signs(annotated, signs)
        annotated = self._draw_telemetry(annotated, speed_map, risk_map)

        max_risk = max((v["risk_pct"] for v in risk_map.values()), default=0.0)

        results = {
            "tracks":       tracks,
            "speeds":       speed_map,
            "ttc":          ttc_map,
            "risk":         risk_map,
            "signs":        signs,
            "max_risk_pct": max_risk,
        }

        return annotated, results

    @staticmethod
    def _classify_signs(frame: np.ndarray, tracks: List[Dict]) -> List[Dict]:
        """Stage 2 of the pipeline: EfficientNet on every sign / light crop."""
        try:
            from main_predict_cnn import classify_sign, is_available
        except Exception:  # torch / checkpoint problems must not kill live detection
            return []
        if not is_available():
            return []

        h, w = frame.shape[:2]
        out = []
        for t in tracks:
            if not any(k in t["cls_name"].lower() for k in SIGN_CLASSES):
                continue
            x1, y1, x2, y2 = map(int, t["bbox"])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            name, conf = classify_sign(frame[y1:y2, x1:x2])
            if name != "UNKNOWN":
                out.append({"bbox": [x1, y1, x2, y2], "name": name, "confidence": round(conf, 1)})
        return out

    @staticmethod
    def _draw_signs(frame: np.ndarray, signs: List[Dict]) -> np.ndarray:
        for s in signs:
            x1, y1 = s["bbox"][0], s["bbox"][1]
            cv2.putText(frame, f"{s['name']} ({s['confidence']:.0f}%)", (x1, max(20, y1 - 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_telemetry(frame: np.ndarray, speed_map: Dict, risk_map: Dict) -> np.ndarray:
        n = len(speed_map)
        if n == 0:
            return frame

        max_risk  = max((v["risk_pct"] for v in risk_map.values()), default=0.0)
        avg_speed = sum(v["speed_kmh"] for v in speed_map.values()) / n

        lines = [
            f"Objects tracked : {n}",
            f"Avg speed       : {avg_speed:.1f} km/h",
            f"Max risk        : {max_risk:.0f} %",
        ]

        y0, dy = 22, 22
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)

        return frame
