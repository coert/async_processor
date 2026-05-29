from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _TrackedMarker:
    marker_id: int
    corners: np.ndarray


@dataclass(frozen=True)
class _TrackingState:
    gray: np.ndarray
    quad: np.ndarray | None
    dictionary_name: str | None


@dataclass(frozen=True)
class _QuadTrackResult:
    corners: np.ndarray | None
    confidence: float | None
    reason: str

    @property
    def succeeded(self) -> bool:
        return self.corners is not None and self.confidence is not None


@dataclass(frozen=True)
class _MarkerTrackDebug:
    marker_id: int
    previous_corners: np.ndarray
    result: _QuadTrackResult


@dataclass(frozen=True)
class _QuadTrackingPlan:
    prior_quad: np.ndarray | None
    quad_result: _QuadTrackResult | None
    refresh_reason: str
    force_full_rectifier: bool
    confidence: float | None
