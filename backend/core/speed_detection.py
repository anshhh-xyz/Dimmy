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

# The default trapezoid above was drawn for a 1280x720 frame.
_DEFAULT_REF_SIZE = (1280, 720)

_DEFAULT_DST_SIZE_M = (6.0, 20.0)
_DEFAULT_DST_PX_PER_M = 40

# Anything further than this is treated as "no meaningful distance". Points
# above the horizon line of the calibration warp to nonsense values otherwise.
_MAX_DISTANCE_M = 150.0

# Speeds computed over less than this many seconds are pure noise (two frames
# arriving a few ms apart turn a 2 px jitter into hundreds of km/h).
_MIN_DT_S = 0.05


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

        # Sign of the homogeneous w inside the calibrated road region. Pixels whose
        # w has the opposite sign are above the horizon (see warp_point).
        cx, cy = src_pts.astype(np.float64).mean(axis=0)
        w_ref = self.M[2, 0] * cx + self.M[2, 1] * cy + self.M[2, 2]
        self._w_sign = 1.0 if w_ref >= 0 else -1.0

    def warp_point(self, px: float, py: float) -> Tuple[float, float]:
        """Pixel -> (x_m, y_m) on the ground plane; (nan, nan) if it cannot be on the road.

        Pixels at or above the horizon of the calibration (the vanishing line)
        get a non-positive homogeneous w. perspectiveTransform silently flips
        those to the wrong side, which made far-away / above-horizon objects
        appear to be RIGHT in front of the car (distance 0 m, 60% risk).
        """
        w = (self.M[2, 0] * px + self.M[2, 1] * py + self.M[2, 2]) * self._w_sign
        if w <= 1e-9:
            return float("nan"), float("nan")
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
        if dt < _MIN_DT_S:
            return 0.0
        dist_m = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        speed_ms = dist_m / dt
        return speed_ms * 3.6

    def closing_speed_kmh(self, height_m: float) -> float:
        """Signed speed at which the object is getting closer (km/h).

        Positive = approaching the camera, negative = moving away. Unlike
        compute_speed() this only looks at the forward axis, so sideways
        motion / bbox jitter does not count as "approaching".
        """
        if len(self.positions_m) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt < _MIN_DT_S:
            return 0.0
        d0 = _distance_from_my(self.positions_m[0][1], height_m)
        d1 = _distance_from_my(self.positions_m[-1][1], height_m)
        return (d0 - d1) / dt * 3.6

    def current_distance_m(self, height_m: float) -> float:
        if not self.positions_m:
            return float("inf")
        return _distance_from_my(self.positions_m[-1][1], height_m)


def _distance_from_my(my: float, height_m: float) -> float:
    """Convert a bird's-eye y coordinate to distance ahead of the camera.

    In the warped image y=0 is the FAR edge of the calibration region and
    y=height_m is the NEAR edge (bottom of the frame). The original code used
    `my` directly as the distance, so objects right in front of the car were
    reported as ~20 m away and distant ones as ~0 m - the exact opposite.
    """
    d = height_m - my
    if d != d:                      # NaN
        return float("inf")
    if d > _MAX_DISTANCE_M:         # too far to be meaningful
        return float("inf")
    return max(d, 0.0)


class SpeedEstimator:

    def __init__(
        self,
        src_pts: Optional[List] = None,
        dst_size_m: Tuple[float, float] = _DEFAULT_DST_SIZE_M,
        smoothing_window: int = 5,
        px_per_m: int = _DEFAULT_DST_PX_PER_M,
    ) -> None:
        self._custom_pts = src_pts is not None
        self._dst_size_m = dst_size_m
        self._px_per_m = px_per_m
        pts = np.float32(src_pts) if src_pts is not None else _DEFAULT_SRC_PTS
        self.ipm = IPMCalibration(pts, dst_size_m, px_per_m)
        self.window = smoothing_window
        self._tracks: Dict[int, _TrackRecord] = {}
        self._frame_size: Optional[Tuple[int, int]] = None

    def set_frame_size(self, width: int, height: int) -> None:
        """Rescale the default calibration trapezoid to the real frame size.

        The default points assume a 1280x720 frame; on a 640x480 webcam they
        landed outside the picture and every distance/speed was garbage.
        Custom calibration points are assumed to already match the frame.
        """
        if self._frame_size == (width, height):
            return
        self._frame_size = (width, height)
        if self._custom_pts:
            return
        sx = width / _DEFAULT_REF_SIZE[0]
        sy = height / _DEFAULT_REF_SIZE[1]
        pts = _DEFAULT_SRC_PTS * np.float32([sx, sy])
        self.ipm = IPMCalibration(pts, self._dst_size_m, self._px_per_m)
        self._tracks.clear()

    def update(
        self,
        tracks: List[Dict],
        fps: float = 30.0,
        timestamp: Optional[float] = None,
    ) -> Dict[int, Dict]:
        """`timestamp` (seconds) should be supplied when processing video files
        (frame_index / fps). Using wall-clock time there made speeds depend on
        how fast the computer happened to process the file."""
        now = time.perf_counter() if timestamp is None else float(timestamp)
        results: Dict[int, Dict] = {}
        height_m = self.ipm.height_m
        tracks = [t for t in tracks if int(t["track_id"]) >= 0]  # untracked boxes have no history

        for t in tracks:
            tid  = int(t["track_id"])
            bbox = t["bbox"]
            cx   = (bbox[0] + bbox[2]) / 2.0
            cy   = float(bbox[3])

            mx, my = self.ipm.warp_point(cx, cy)

            if mx != mx or my != my:  # above the horizon: no usable ground position
                results[tid] = {"speed_kmh": 0.0, "distance_m": float("inf"), "closing_kmh": 0.0}
                self._tracks.pop(tid, None)
                continue

            if tid not in self._tracks:
                self._tracks[tid] = _TrackRecord(self.window)

            rec = self._tracks[tid]
            rec.add(mx, my, now)

            speed_kmh  = rec.compute_speed()
            distance_m = rec.current_distance_m(height_m)
            closing    = rec.closing_speed_kmh(height_m)

            results[tid] = {
                "speed_kmh":   round(speed_kmh,  2),
                "distance_m":  round(distance_m, 2),
                "closing_kmh": round(closing, 2),
            }

        active_ids = {int(t["track_id"]) for t in tracks}  # tracks not seen this frame are dropped
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

        return results

    def draw_bev(self, frame: np.ndarray) -> np.ndarray:
        return self.ipm.warp_frame(frame)

    def reset(self) -> None:
        self._tracks.clear()
