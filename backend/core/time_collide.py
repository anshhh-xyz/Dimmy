from __future__ import annotations

from typing import Dict, Optional

_INF_TTC = float("inf")

TTC_SAFE_THRESHOLD    = 5.0
TTC_CAUTION_THRESHOLD = 2.5


class TTCEngine:

    def __init__(
        self,
        ego_speed_kmh: float = 0.0,
        min_approach_speed_kmh: float = 1.0,
    ) -> None:
        self.default_ego_speed_kmh = ego_speed_kmh
        self.min_approach_speed_kmh = min_approach_speed_kmh

    def update(
        self,
        speed_map: Dict[int, Dict],
        ego_speed_kmh: Optional[float] = None,
    ) -> Dict[int, Dict]:
        ego_kmh = ego_speed_kmh if ego_speed_kmh is not None else self.default_ego_speed_kmh
        results: Dict[int, Dict] = {}

        for tid, data in speed_map.items():
            target_speed_kmh: float = data.get("speed_kmh", 0.0)
            distance_m:       float = data.get("distance_m", _INF_TTC)

            rel_speed_kmh = ego_kmh - target_speed_kmh
            rel_speed_ms  = rel_speed_kmh / 3.6

            if rel_speed_kmh < self.min_approach_speed_kmh or distance_m == _INF_TTC:
                ttc_s = _INF_TTC
            else:
                ttc_s = distance_m / rel_speed_ms

            results[tid] = {
                "ttc_s":              round(ttc_s, 3) if ttc_s != _INF_TTC else _INF_TTC,
                "distance_m":         round(distance_m, 2),
                "relative_speed_kmh": round(rel_speed_kmh, 2),
                "relative_speed_ms":  round(rel_speed_ms, 4),
            }

        return results

    @staticmethod
    def compute_single(
        distance_m: float,
        relative_speed_ms: float,
    ) -> float:
        if relative_speed_ms <= 0:
            return _INF_TTC
        return distance_m / relative_speed_ms
