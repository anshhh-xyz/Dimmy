from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracker               import ByteTracker
from speed_detection       import SpeedEstimator
from time_collide          import TTCEngine
from probability_calculator import AccidentProbabilityCalculator, draw_risk_overlay


class ADASPipeline:

    def __init__(
        self,
        model_path:       str,
        ipm_src_pts:      Optional[List] = None,
        ipm_dst_size_m:   Tuple[float, float] = (6.0, 20.0),
        ego_speed_kmh:    float = 0.0,
        smoothing_window: int   = 5,
        device:           str   = "cpu",
    ) -> None:
        self.tracker   = ByteTracker(model_path, device=device)
        self.speed_est = SpeedEstimator(
            src_pts          = ipm_src_pts,
            dst_size_m       = ipm_dst_size_m,
            smoothing_window = smoothing_window,
        )
        self.ttc_engine    = TTCEngine()
        self.risk_calc     = AccidentProbabilityCalculator()
        self.ego_speed_kmh = ego_speed_kmh

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        tracks     = self.tracker.update(frame)
        speed_map  = self.speed_est.update(tracks)
        ttc_map    = self.ttc_engine.update(speed_map, ego_speed_kmh=self.ego_speed_kmh)
        risk_map   = self.risk_calc.compute(ttc_map)

        bbox_map: Dict[int, list] = {
            int(t["track_id"]): t["bbox"] for t in tracks
        }

        annotated = self.tracker.annotate(frame, tracks)
        annotated = draw_risk_overlay(annotated, bbox_map, risk_map)
        annotated = self._draw_telemetry(annotated, speed_map, risk_map)

        max_risk = max((v["risk_pct"] for v in risk_map.values()), default=0.0)

        results = {
            "tracks":       tracks,
            "speeds":       speed_map,
            "ttc":          ttc_map,
            "risk":         risk_map,
            "max_risk_pct": max_risk,
        }

        return annotated, results

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
