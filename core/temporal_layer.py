"""Temporal smoothing utilities for frame-level scores."""

from __future__ import annotations

from collections import deque
from typing import Deque


class TemporalLayer:
    """Maintains a rolling window and returns recency-weighted averages."""

    def __init__(self, window_size: int = 30, ema_alpha: float = 0.65) -> None:
        self.window_size = window_size
        self._scores: Deque[float] = deque(maxlen=window_size)
        self._ema: float | None = None
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))

    def update(self, value: float) -> float:
        score = float(value)
        self._scores.append(score)

        if self._ema is None:
            self._ema = score
            return score

        self._ema = self._ema_alpha * score + (1.0 - self._ema_alpha) * self._ema
        return self._ema