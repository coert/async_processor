from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np


def order_corners(pts: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(pts, dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()

    tl = points[int(np.argmin(s))]
    br = points[int(np.argmax(s))]
    tr = points[int(np.argmin(diff))]
    bl = points[int(np.argmax(diff))]

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    if len({tuple(map(float, p)) for p in ordered}) != 4:
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]
        start = int(np.argmin(points.sum(axis=1)))
        ordered = np.roll(points, -start, axis=0).astype(np.float32)
        if signed_area(ordered) < 0:
            ordered = np.array(
                [ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32
            )
    return ordered


def signed_area(quad: np.ndarray) -> float:
    pts = np.asarray(quad, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(quad: np.ndarray) -> float:
    return abs(signed_area(quad))


def is_convex_quad(quad: np.ndarray) -> bool:
    quad = np.asarray(quad, dtype=np.float32)
    crosses = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        c = quad[(i + 2) % 4]
        ab = b - a
        bc = c - b
        crosses.append(ab[0] * bc[1] - ab[1] * bc[0])
    crosses = np.asarray(crosses)
    return bool(np.all(crosses > 0) or np.all(crosses < 0))


def edge_lengths(quad: np.ndarray) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32)
    return np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)


def is_valid_quad(quad: np.ndarray, width: int, height: int, min_area: float) -> bool:
    quad = order_corners(quad)
    if quad.shape != (4, 2):
        return False
    if not np.all(np.isfinite(quad)):
        return False
    if np.any(quad[:, 0] < -1) or np.any(quad[:, 0] > width):
        return False
    if np.any(quad[:, 1] < -1) or np.any(quad[:, 1] > height):
        return False
    if polygon_area(quad) < min_area:
        return False
    if not is_convex_quad(quad):
        return False
    lengths = edge_lengths(quad)
    if np.min(lengths) < 8:
        return False
    skinny = polygon_area(quad) / max(float(np.sum(lengths) ** 2), 1.0)
    if skinny < 0.005:
        return False
    return True


def quad_iou(quad1: np.ndarray, quad2: np.ndarray) -> float:
    quad1 = order_corners(quad1).astype(np.float32)
    quad2 = order_corners(quad2).astype(np.float32)
    area1 = polygon_area(quad1)
    area2 = polygon_area(quad2)
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(quad1, quad2)
    if inter_area <= 0.0:
        return 0.0

    union_area = area1 + area2 - float(inter_area)
    if union_area <= 0.0:
        return 0.0
    return float(inter_area) / union_area


def dedupe_quads(quads: Iterable[np.ndarray], tol: float = 10.0) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    binned_quads: dict[tuple[int, int], list[np.ndarray]] = {}
    cell_size = tol if tol > 0.0 else 1.0

    for quad in quads:
        quad = order_corners(quad)
        centroid = np.mean(quad, axis=0)
        cell = (
            int(np.floor(float(centroid[0]) / cell_size)),
            int(np.floor(float(centroid[1]) / cell_size)),
        )

        is_duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in binned_quads.get((cell[0] + dx, cell[1] + dy), []):
                    if np.mean(np.linalg.norm(quad - other, axis=1)) < tol:
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break

        if is_duplicate:
            continue

        unique.append(quad)
        binned_quads.setdefault(cell, []).append(quad)

    return unique
