from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


TTC_SAFE     = 5.0
TTC_CAUTION  = 2.5

RISK_SAFE_MIN     = 0.0
RISK_SAFE_MAX     = 20.0
RISK_CAUTION_MIN  = 20.0
RISK_CAUTION_MAX  = 70.0
RISK_CRITICAL_MIN = 70.0
RISK_CRITICAL_MAX = 100.0

PROXIMITY_FLOOR_DIST_M    = 3.0
PROXIMITY_FLOOR_RISK_PCT  = 60.0
HIGH_SPEED_THRESHOLD_KMH  = 80.0


class RiskLevel(str, Enum):
    SAFE     = "SAFE"
    CAUTION  = "CAUTION"
    CRITICAL = "CRITICAL"
    UNKNOWN  = "UNKNOWN"


@dataclass
class RiskResult:
    track_id:           int
    risk_pct:           float
    risk_level:         RiskLevel
    ttc_s:              float
    distance_m:         float
    relative_speed_kmh: float
    label:              str


def _lerp(value: float, in_min: float, in_max: float,
          out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return out_min
    t = (value - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


def _ttc_to_base_risk(ttc_s: float) -> tuple[float, RiskLevel]:
    if ttc_s == math.inf or ttc_s < 0:
        return 0.0, RiskLevel.UNKNOWN

    if ttc_s > TTC_SAFE:
        risk = _lerp(ttc_s, TTC_SAFE, TTC_SAFE * 3, RISK_SAFE_MAX, RISK_SAFE_MIN)
        return risk, RiskLevel.SAFE

    if ttc_s > TTC_CAUTION:
        risk = _lerp(ttc_s, TTC_CAUTION, TTC_SAFE, RISK_CAUTION_MAX, RISK_CAUTION_MIN)
        return risk, RiskLevel.CAUTION

    risk = _lerp(ttc_s, 0.0, TTC_CAUTION, RISK_CRITICAL_MAX, RISK_CRITICAL_MIN)
    return risk, RiskLevel.CRITICAL


class AccidentProbabilityCalculator:

    def __init__(
        self,
        proximity_floor: bool = True,
        speed_boost: bool = True,
        proximity_floor_dist_m: float = PROXIMITY_FLOOR_DIST_M,
        proximity_floor_risk_pct: float = PROXIMITY_FLOOR_RISK_PCT,
        high_speed_threshold_kmh: float = HIGH_SPEED_THRESHOLD_KMH,
    ) -> None:
        self.proximity_floor      = proximity_floor
        self.speed_boost          = speed_boost
        self.floor_dist_m         = proximity_floor_dist_m
        self.floor_risk_pct       = proximity_floor_risk_pct
        self.high_speed_threshold = high_speed_threshold_kmh

    def compute(self, ttc_results: Dict[int, Dict]) -> Dict[int, Dict]:
        output: Dict[int, Dict] = {}
        for tid, data in ttc_results.items():
            res = self._score(
                track_id      = tid,
                ttc_s         = data.get("ttc_s", math.inf),
                distance_m    = data.get("distance_m", math.inf),
                rel_speed_kmh = data.get("relative_speed_kmh", 0.0),
            )
            output[tid] = {
                "risk_pct":           round(res.risk_pct, 1),
                "risk_level":         res.risk_level.value,
                "label":              res.label,
                "ttc_s":              res.ttc_s,
                "distance_m":         res.distance_m,
                "relative_speed_kmh": res.relative_speed_kmh,
            }
        return output

    def compute_single(
        self,
        ttc_s: float,
        distance_m: float,
        rel_speed_kmh: float = 0.0,
        track_id: int = -1,
    ) -> Dict:
        res = self._score(track_id, ttc_s, distance_m, rel_speed_kmh)
        return {
            "risk_pct":           round(res.risk_pct, 1),
            "risk_level":         res.risk_level.value,
            "label":              res.label,
            "ttc_s":              res.ttc_s,
            "distance_m":         res.distance_m,
            "relative_speed_kmh": res.relative_speed_kmh,
        }

    def _score(
        self,
        track_id:      int,
        ttc_s:         float,
        distance_m:    float,
        rel_speed_kmh: float,
    ) -> RiskResult:
        risk_pct, level = _ttc_to_base_risk(ttc_s)

        if self.proximity_floor and distance_m < self.floor_dist_m:
            if risk_pct < self.floor_risk_pct:
                risk_pct = self.floor_risk_pct
                if level in (RiskLevel.SAFE, RiskLevel.UNKNOWN):
                    level = RiskLevel.CAUTION

        if self.speed_boost and rel_speed_kmh > self.high_speed_threshold:
            excess_ratio = min(
                (rel_speed_kmh - self.high_speed_threshold) / self.high_speed_threshold,
                1.0,
            )
            boost = excess_ratio * (RISK_CRITICAL_MAX - risk_pct)
            risk_pct = min(risk_pct + boost, RISK_CRITICAL_MAX)
            if risk_pct >= RISK_CRITICAL_MIN:
                level = RiskLevel.CRITICAL
            elif risk_pct >= RISK_CAUTION_MIN:
                level = RiskLevel.CAUTION

        risk_pct = round(min(max(risk_pct, 0.0), 100.0), 2)

        if level == RiskLevel.UNKNOWN:
            ttc_str = "No threat"
        elif ttc_s == math.inf:
            ttc_str = "inf"
        else:
            ttc_str = f"{ttc_s:.2f}s"

        label = f"[{level.value}] {risk_pct:.0f}% risk | TTC={ttc_str} | {distance_m:.1f}m @ {rel_speed_kmh:.1f}km/h"

        return RiskResult(
            track_id           = track_id,
            risk_pct           = risk_pct,
            risk_level         = level,
            ttc_s              = ttc_s,
            distance_m         = distance_m,
            relative_speed_kmh = rel_speed_kmh,
            label              = label,
        )


_LEVEL_COLOURS = {
    RiskLevel.SAFE:     (0,   220, 60),
    RiskLevel.CAUTION:  (0,   200, 255),
    RiskLevel.CRITICAL: (0,   30,  255),
    RiskLevel.UNKNOWN:  (180, 180, 180),
}

try:
    import cv2

    def draw_risk_overlay(frame, bbox_map: Dict[int, list], risk_map: Dict[int, Dict]):
        import math as _math
        for tid, data in risk_map.items():
            bbox = bbox_map.get(tid)
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            level  = RiskLevel(data["risk_level"])
            colour = _LEVEL_COLOURS.get(level, (200, 200, 200))
            pct    = data["risk_pct"]
            ttc    = data["ttc_s"]
            ttc_str = f"{ttc:.1f}s" if ttc != _math.inf else "N/A"
            badge_text = f"{level.value} {pct:.0f}% TTC={ttc_str}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by = max(y1 - 4, th + 4)
            cv2.rectangle(frame, (x1, by - th - baseline - 2), (x1 + tw + 4, by + 2), colour, cv2.FILLED)
            cv2.putText(frame, badge_text, (x1 + 2, by - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

except ImportError:
    def draw_risk_overlay(frame, bbox_map, risk_map):
        return frame
