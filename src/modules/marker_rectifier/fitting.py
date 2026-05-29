from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .geometry import (
    edge_lengths,
    is_convex_quad,
    is_valid_quad,
    order_corners,
    polygon_area,
)
from .models import Candidate


def sample_edge_points(quad: np.ndarray, n_per_edge: int = 100) -> np.ndarray:
    points = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        ts = np.linspace(0.0, 1.0, n_per_edge, endpoint=True)
        seg = (1.0 - ts)[:, None] * p0 + ts[:, None] * p1
        points.append(seg)
    return np.vstack(points).astype(np.float32)


def bilinear_sample(
    image: np.ndarray, points: np.ndarray, outside_value: float
) -> np.ndarray:
    h, w = image.shape[:2]
    x = points[:, 0]
    y = points[:, 1]

    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    values = np.full(points.shape[0], outside_value, dtype=np.float32)
    if not np.any(valid):
        return values

    xv = x[valid]
    yv = y[valid]
    x0 = np.floor(xv).astype(np.int32)
    y0 = np.floor(yv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = xv - x0
    wy = yv - y0

    sampled = (
        (1 - wx) * (1 - wy) * image[y0, x0]
        + wx * (1 - wy) * image[y0, x1]
        + (1 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )
    values[valid] = sampled.astype(np.float32)
    return values


def rectification_penalty(quad: np.ndarray) -> float:
    lengths = edge_lengths(quad)
    if np.min(lengths) < 1e-3:
        return 1e3
    ratio = float(np.max(lengths) / np.min(lengths))
    diag = np.linalg.norm(quad[0] - quad[2]) + np.linalg.norm(quad[1] - quad[3])
    area = polygon_area(quad)
    compactness = area / max(diag * diag, 1.0)
    penalty = max(0.0, ratio - 8.0) * 0.5
    penalty += max(0.0, 0.08 - compactness) * 20.0
    return float(penalty)


def edge_distance_score(
    quad: np.ndarray, dist: np.ndarray, grad_mag: np.ndarray, n: int = 100
) -> float:
    points = sample_edge_points(quad, n_per_edge=n)
    dist_values = bilinear_sample(dist, points, outside_value=999.0)
    grad_values = bilinear_sample(grad_mag, points, outside_value=0.0)
    mean_dist = float(np.mean(dist_values))
    grad_penalty = max(0.0, 18.0 - float(np.mean(grad_values))) * 0.1
    return mean_dist + grad_penalty + rectification_penalty(quad)


def candidate_selection_score(
    candidate: Candidate,
    edge_score: float,
    width: int,
    height: int,
    bw_dominance: float = 0.5,
) -> float:
    image_area = max(float(width * height), 1.0)
    area_ratio = min(polygon_area(candidate.quad) / image_area, 0.35)
    score = edge_score - bw_dominance * 120.0 - area_ratio * 90.0

    if candidate.source != "hough_dominant":
        return score

    lengths = edge_lengths(candidate.quad)
    side_ratio = float(np.max(lengths) / max(np.min(lengths), 1e-6))
    stripe_penalty = max(0.0, side_ratio - 4.0) * 2.0
    return score - 90.0 * area_ratio + stripe_penalty


def refine_candidate(
    initial_quad: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    min_area: float,
    reg_weight: float = 0.05,
) -> np.ndarray:
    initial_quad = order_corners(initial_quad).astype(np.float32)
    initial_quad[:, 0] = np.clip(initial_quad[:, 0], 0.0, width - 1.0)
    initial_quad[:, 1] = np.clip(initial_quad[:, 1], 0.0, height - 1.0)
    initial_quad = order_corners(initial_quad).astype(np.float32)
    sample_t = np.linspace(0.0, 1.0, 80, endpoint=True)

    def residuals(params: np.ndarray) -> np.ndarray:
        quad = order_corners(params.reshape(4, 2))
        edge_points = []
        for i in range(4):
            p0 = quad[i]
            p1 = quad[(i + 1) % 4]
            edge_points.append((1.0 - sample_t)[:, None] * p0 + sample_t[:, None] * p1)
        edge_points_arr = np.vstack(edge_points).astype(np.float32)
        residual = list(bilinear_sample(dist, edge_points_arr, outside_value=60.0))

        area = polygon_area(quad)
        lengths = edge_lengths(quad)
        convex_penalty = 50.0 if not is_convex_quad(quad) else 0.0
        area_penalty = math.sqrt(max(min_area - area, 0.0))
        length_penalty = 20.0 if np.min(lengths) < 8.0 else 0.0
        residual.extend([convex_penalty] * 12)
        residual.extend([area_penalty] * 12)
        residual.extend([length_penalty] * 8)

        residual.extend(((quad - initial_quad).reshape(-1) * reg_weight).tolist())
        return np.asarray(residual, dtype=np.float32)

    lower = np.tile([0.0, 0.0], 4)
    upper = np.tile([width - 1.0, height - 1.0], 4)
    result = least_squares(
        residuals,
        initial_quad.reshape(-1),
        bounds=(lower, upper),
        max_nfev=200,
    )
    refined = order_corners(result.x.reshape(4, 2))
    if not is_valid_quad(refined, width, height, min_area):
        return initial_quad
    return refined.astype(np.float32)
