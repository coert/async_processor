from __future__ import annotations

import cv2
import numpy as np

from .constants import (
    BW_BLACK_THRESHOLD,
    BW_NEUTRAL_CHROMA_THRESHOLD,
    BW_PRIORITY_THRESHOLD,
    BW_WHITE_THRESHOLD,
    EPSILONS,
    NMS_IOU_THRESHOLD,
)
from .geometry import dedupe_quads, is_valid_quad, order_corners, polygon_area, quad_iou
from .models import Candidate
from .perspective import warp_square_cutout


def apply_nms(
    candidates: list[Candidate], iou_threshold: float = NMS_IOU_THRESHOLD
) -> list[Candidate]:
    sorted_cands = sorted(
        candidates,
        key=lambda candidate: (
            -float((candidate.bw_dominance or 0.0) >= BW_PRIORITY_THRESHOLD),
            -polygon_area(candidate.quad),
            -(candidate.bw_dominance or 0.0),
            candidate.score if candidate.score is not None else float("inf"),
        ),
    )
    keep: list[Candidate] = []

    for candidate in sorted_cands:
        if any(quad_iou(candidate.quad, kept.quad) >= iou_threshold for kept in keep):
            continue
        keep.append(candidate)
    return keep


def compute_bw_dominance(
    image: np.ndarray, quad: np.ndarray, out_size: int = 64
) -> float:
    cutout = warp_square_cutout(image, quad, out_size)
    lab = cv2.cvtColor(cutout, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:].astype(np.float32) - 128.0, axis=2)

    neutral_mask = chroma <= BW_NEUTRAL_CHROMA_THRESHOLD
    black_mask = lightness <= BW_BLACK_THRESHOLD
    white_mask = lightness >= BW_WHITE_THRESHOLD
    bw_pixels = np.count_nonzero(neutral_mask & (black_mask | white_mask))
    return float(bw_pixels) / float(out_size * out_size)


def find_contour_candidates(
    edges: np.ndarray, width: int, height: int, min_area: float, variant_idx: int
) -> list[Candidate]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        for eps in EPSILONS:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) != 4:
                continue
            approx = approx.reshape(-1, 2).astype(np.float32)
            if cv2.contourArea(approx) < min_area:
                continue
            if not cv2.isContourConvex(approx.astype(np.int32)):
                continue
            if is_valid_quad(approx, width, height, min_area):
                candidates.append(order_corners(approx))

    return [
        Candidate(quad=quad, source="contour", variant_idx=variant_idx)
        for quad in dedupe_quads(candidates)
    ]
