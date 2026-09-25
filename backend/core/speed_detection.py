from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_DEFAULT_SRC_PTS = np.float32([
    [540, 450],
    [740, 450],
    [1100, 680],
    [180, 680],
])

_DEFAULT_DST_SIZE_M = (6.0, 20.0)
_DEFAULT_DST_PX_PER_M = 40


class IPMCalibration:

    def __init__(
        self,
        src_pts: np.ndarray,
        dst_size_m: Tuple[float, float],
        px_per_m: int = _DEFAULT_DST_PX_PER_M,
    ) -> None:
        self.width_m, self.height_m = dst_size_m
        self.px_per_m = px_per_m

        dst_w = int(self.width_m * px_per_m)
        dst_h = int(self.height_m * px_per_m)

        dst_pts = np.float32([
            [0,     0    ],
            [dst_w, 0    ],
            [dst_w, dst_h],
            [0,     dst_h],
        ])

        self.M: np.ndarray = cv2.getPerspectiveTransform(
            src_pts.astype(np.float32), dst_pts
        )
        self.dst_shape: Tuple[int, int] = (dst_w, dst_h)

    def warp_point(self, px: float, py: float) -> Tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.M)[0][0]
        mx = warped[0] / self.px_per_m
        my = warped[1] / self.px_per_m
        return float(mx), float(my)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M, self.dst_shape)


class _TrackRecord:

    def __init__(self, window: int) -> None:
        self.positions_m: deque[Tuple[float, float]] = deque(maxlen=window)
        self.timestamps:  deque[float]               = deque(maxlen=window)

    def add(self, mx: float, my: float, ts: float) -> None:
        self.positions_m.append((mx, my))
        self.timestamps.append(ts)

    def compute_speed(self) -> float:
        if len(self.positions_m) < 2:
            return 0.0
        x0, y0 = self.positions_m[0]
        x1, y1 = self.positions_m[-1]
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0.0:
            return 0.0
        dist_m = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        speed_ms = dist_m / dt
        return speed_ms * 3.6

    def current_distance_m(self) -> float:
        if not self.positions_m:
            return float("inf")
        _, my = self.positions_m[-1]
        return max(0.0, float(my))


class SpeedEstimator:

    def __init__(
        self,
        src_pts: Optional[List] = None,
        dst_size_m: Tuple[float, float] = _DEFAULT_DST_SIZE_M,
        smoothing_window: int = 5,
        px_per_m: int = _DEFAULT_DST_PX_PER_M,
    ) -> None:
        pts = np.float32(src_pts) if src_pts is not None else _DEFAULT_SRC_PTS
        self.ipm = IPMCalibration(pts, dst_size_m, px_per_m)
        self.window = smoothing_window
        self._tracks: Dict[int, _TrackRecord] = {}

    def update(
        self,
        tracks: List[Dict],
        fps: float = 30.0,
    ) -> Dict[int, Dict]:
        now = time.perf_counter()
        results: Dict[int, Dict] = {}

        for t in tracks:
            tid  = int(t["track_id"])
            bbox = t["bbox"]
            cx   = (bbox[0] + bbox[2]) / 2.0
            cy   = float(bbox[3])

            mx, my = self.ipm.warp_point(cx, cy)

            if tid not in self._tracks:
                self._tracks[tid] = _TrackRecord(self.window)

            rec = self._tracks[tid]
            rec.add(mx, my, now)

            speed_kmh  = rec.compute_speed()
            distance_m = rec.current_distance_m()

            results[tid] = {
                "speed_kmh":  round(speed_kmh,  2),
                "distance_m": round(distance_m, 2),
            }

        active_ids = {int(t["track_id"]) for t in tracks}
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

        return results

    def draw_bev(self, frame: np.ndarray) -> np.ndarray:
        return self.ipm.warp_frame(frame)

    def reset(self) -> None:
        self._tracks.clear()
