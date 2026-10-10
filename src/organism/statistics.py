from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _require_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(slots=True)
class PairAccumulator:
    """Bounded online bivariate statistics used by actuator effect evidence."""

    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    c_xy: float = 0.0
    _corr_valid: bool = field(default=False, init=False, repr=False, compare=False)
    _cached_corr: float | None = field(default=None, init=False, repr=False, compare=False)

    def observe(self, x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            return
        self._corr_valid = False
        self.count += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.count
        dy = y - self.mean_y
        self.mean_y += dy / self.count
        self.c_xy += dx * (y - self.mean_y)
        self.m2_x += dx * (x - self.mean_x)
        self.m2_y += dy * (y - self.mean_y)

    @property
    def correlation(self) -> float | None:
        if self._corr_valid:
            return self._cached_corr
        if self.count < 3 or self.m2_x <= 1e-18 or self.m2_y <= 1e-18:
            result = None
        else:
            value = self.c_xy / math.sqrt(self.m2_x * self.m2_y)
            result = max(-1.0, min(1.0, value)) if math.isfinite(value) else None
        self._cached_corr = result
        self._corr_valid = True
        return result

    def to_payload(self) -> dict[str, float | int]:
        return {"count": self.count, "mean_x": self.mean_x, "mean_y": self.mean_y, "m2_x": self.m2_x, "m2_y": self.m2_y, "c_xy": self.c_xy}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PairAccumulator":
        required = {"count", "mean_x", "mean_y", "m2_x", "m2_y", "c_xy"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("pair accumulator checkpoint fields do not match schema")
        count = _require_nonneg_int(payload["count"], "count")
        mean_x = _require_finite(payload["mean_x"], "mean_x")
        mean_y = _require_finite(payload["mean_y"], "mean_y")
        m2_x = _require_finite(payload["m2_x"], "m2_x")
        m2_y = _require_finite(payload["m2_y"], "m2_y")
        if m2_x < 0 or m2_y < 0:
            raise ValueError("m2_x and m2_y must be non-negative")
        c_xy = _require_finite(payload["c_xy"], "c_xy")
        return cls(count=count, mean_x=mean_x, mean_y=mean_y, m2_x=m2_x, m2_y=m2_y, c_xy=c_xy)
