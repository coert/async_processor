from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeArtifacts:
    gray: np.ndarray
    blur: np.ndarray
    edges_canny: np.ndarray
    grad_mag: np.ndarray
    dist: np.ndarray


@dataclass
class Candidate:
    quad: np.ndarray
    source: str
    variant_idx: int
    score: float | None = None
    bw_dominance: float | None = None


@dataclass
class FitResult:
    quad: np.ndarray
    score: float
    rejected: list[np.ndarray]


@dataclass(frozen=True)
class MarkerEvidence:
    detected_count: int
    rejected_count: int


@dataclass
class LineDebug:
    variant_idx: int
    lines: np.ndarray | None
    family_a: np.ndarray | None
    family_b: np.ndarray | None
    candidates: list[np.ndarray]
    closed_edges_used: bool = False
