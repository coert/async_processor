from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from .constants import HOUGH_FAMILY_SAMPLE_COUNT, HOUGH_QUAD_CAP
from .geometry import dedupe_quads, is_valid_quad, order_corners, polygon_area
from .models import Candidate, LineDebug


def cap_quads_by_area(
    quads: list[np.ndarray], cap: int = HOUGH_QUAD_CAP
) -> list[np.ndarray]:
    if len(quads) <= cap:
        return quads

    ordered = sorted(quads, key=polygon_area, reverse=True)
    positions = np.linspace(0, len(ordered) - 1, num=cap, dtype=int)
    sampled: list[np.ndarray] = []
    seen_positions: set[int] = set()
    for position in positions:
        position_int = int(position)
        if position_int in seen_positions:
            continue
        seen_positions.add(position_int)
        sampled.append(ordered[position_int])
    return sampled


def line_to_abc(line: Sequence[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = map(float, line)
    a = y1 - y2
    b = x2 - x1
    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None
    c = x1 * y2 - x2 * y1
    return np.array([a / norm, b / norm, c / norm], dtype=np.float32)


def line_angle_deg(line: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return angle


def select_angle_families(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    angles = np.array([line_angle_deg(line) for line in lines], dtype=np.float32)
    if len(angles) < 4:
        return None

    bins = np.linspace(0.0, 180.0, 37)
    hist, _ = np.histogram(angles, bins=bins)
    if hist.max() == 0:
        return None

    first_idx = int(np.argmax(hist))
    first_center = (bins[first_idx] + bins[first_idx + 1]) / 2.0

    distances = np.abs(((angles - first_center + 90.0) % 180.0) - 90.0)
    second_mask = distances > 20.0
    if not np.any(second_mask):
        return None

    second_angles = angles[second_mask]
    second_hist, _ = np.histogram(second_angles, bins=bins)
    second_idx = int(np.argmax(second_hist))
    second_center = (bins[second_idx] + bins[second_idx + 1]) / 2.0

    family_a_mask = np.abs(((angles - first_center + 90.0) % 180.0) - 90.0) <= 15.0
    family_b_mask = np.abs(((angles - second_center + 90.0) % 180.0) - 90.0) <= 15.0

    family_a = lines[family_a_mask]
    family_b = lines[family_b_mask]
    if len(family_a) < 2 or len(family_b) < 2:
        return None
    return family_a, family_b


def pair_extreme_lines(lines: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered_lines: list[tuple[float, np.ndarray]] = []
    oriented_directions: list[np.ndarray] = []

    for line in lines.astype(np.float32):
        x1, y1, x2, y2 = map(float, line)
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        if direction[1] > 0.0 or (
            abs(float(direction[1])) < 1e-6 and direction[0] < 0.0
        ):
            direction = -direction
        oriented_directions.append(direction)
        abc = line_to_abc(line)
        if abc is None:
            continue
        ordered_lines.append((0.0, abc))

    if len(ordered_lines) < 2 or len(oriented_directions) < 2:
        return []

    axis_u = np.mean(np.asarray(oriented_directions, dtype=np.float32), axis=0)
    axis_norm = float(np.linalg.norm(axis_u))
    if axis_norm < 1e-6:
        axis_u = oriented_directions[0]
    else:
        axis_u = axis_u / axis_norm
    axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)

    ranked_lines: list[tuple[float, np.ndarray]] = []
    for line in lines.astype(np.float32):
        abc = line_to_abc(line)
        if abc is None:
            continue
        x1, y1, x2, y2 = map(float, line)
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        ranked_lines.append((float(np.dot(midpoint, axis_v)), abc))

    if len(ranked_lines) < 2:
        return []

    ranked_lines.sort(key=lambda item: item[0])
    sample_count = min(HOUGH_FAMILY_SAMPLE_COUNT, len(ranked_lines))
    sample_positions = np.linspace(
        0, len(ranked_lines) - 1, num=sample_count, dtype=int
    )
    sampled_lines: list[np.ndarray] = []
    seen_positions: set[int] = set()
    for position in sample_positions:
        position_int = int(position)
        if position_int in seen_positions:
            continue
        seen_positions.add(position_int)
        sampled_lines.append(ranked_lines[position_int][1])

    candidate_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for first_idx, first_line in enumerate(sampled_lines):
        for second_line in sampled_lines[first_idx + 1 :]:
            if np.allclose(first_line, second_line):
                continue
            candidate_pairs.append((first_line, second_line))
    return candidate_pairs


def intersect_lines(line1: np.ndarray, line2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(float(det)) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def dominant_family_quads(
    edges: np.ndarray,
    family_lines: np.ndarray | None,
    width: int,
    height: int,
    min_area: float,
) -> list[np.ndarray]:
    if family_lines is None or len(family_lines) < 2:
        return []

    directions = []
    endpoints = []
    for x1, y1, x2, y2 in family_lines.astype(np.float32):
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        if direction[1] > 0:
            direction = -direction
        directions.append(direction)
        endpoints.extend(
            [np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)]
        )

    if len(directions) < 2 or len(endpoints) < 4:
        return []

    axis_u = np.mean(np.asarray(directions, dtype=np.float32), axis=0)
    axis_norm = float(np.linalg.norm(axis_u))
    if axis_norm < 1e-6:
        return []
    axis_u = axis_u / axis_norm
    axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)

    endpoints = np.asarray(endpoints, dtype=np.float32)
    endpoint_u = endpoints @ axis_u
    endpoint_v = endpoints @ axis_v

    ys, xs = np.nonzero(edges > 0)
    if len(xs) < 20:
        return []
    edge_points = np.column_stack([xs, ys]).astype(np.float32)
    edge_u = edge_points @ axis_u
    edge_v = edge_points @ axis_v

    quads: list[np.ndarray] = []
    min_dim = float(min(width, height))
    u_bounds = np.percentile(endpoint_u, [0, 100])

    line_infos: list[tuple[float, np.ndarray]] = []
    for line in family_lines.astype(np.float32):
        abc = line_to_abc(line)
        if abc is None:
            continue
        if float(np.dot(abc[:2], axis_v)) < 0.0:
            abc = -abc
        x1, y1, x2, y2 = line
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        line_infos.append((float(midpoint @ axis_v), abc.astype(np.float32)))

    line_infos.sort(key=lambda item: item[0])
    line_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if len(line_infos) >= 2:
        line_pairs.append((line_infos[0][1], line_infos[-1][1]))
    if len(line_infos) >= 4:
        line_pairs.append((line_infos[0][1], line_infos[-2][1]))
        line_pairs.append((line_infos[1][1], line_infos[-1][1]))

    for side0, side1 in line_pairs:
        side_offsets = sorted([-float(side0[2]), -float(side1[2])])
        side_mask = (edge_v >= side_offsets[0] - 40) & (edge_v <= side_offsets[1] + 40)
        if int(np.count_nonzero(side_mask)) < 20:
            continue
        for low_pct, high_pct in [(2, 98), (5, 95), (0, 100)]:
            u_low, u_high = np.percentile(edge_u[side_mask], [low_pct, high_pct])
            if abs(float(u_high - u_low)) < min_dim * 0.08:
                continue
            cross0 = np.array([axis_u[0], axis_u[1], -u_low], dtype=np.float32)
            cross1 = np.array([axis_u[0], axis_u[1], -u_high], dtype=np.float32)
            pts = [
                intersect_lines(side0, cross0),
                intersect_lines(side1, cross0),
                intersect_lines(side1, cross1),
                intersect_lines(side0, cross1),
            ]
            if any(p is None for p in pts):
                continue
            quad = order_corners(np.asarray(pts, dtype=np.float32))
            if is_valid_quad(quad, width, height, min_area):
                quads.append(quad)

    for low_pct, high_pct in [(5, 95), (0, 100), (10, 90)]:
        v_low, v_high = np.percentile(endpoint_v, [low_pct, high_pct])
        if abs(float(v_high - v_low)) < min_dim * 0.08:
            continue

        mask = (
            (edge_v >= v_low - 30)
            & (edge_v <= v_high + 30)
            & (edge_u >= u_bounds[0] - min_dim * 0.15)
            & (edge_u <= u_bounds[1] + min_dim * 0.15)
        )
        if int(np.count_nonzero(mask)) < 20:
            continue

        u_low, u_high = np.percentile(edge_u[mask], [2, 98])
        if abs(float(u_high - u_low)) < min_dim * 0.08:
            continue

        quad_uv = np.array(
            [
                [u_low, v_low],
                [u_high, v_low],
                [u_high, v_high],
                [u_low, v_high],
            ],
            dtype=np.float32,
        )
        quad = quad_uv[:, 0, None] * axis_u + quad_uv[:, 1, None] * axis_v
        quad = order_corners(quad)
        if is_valid_quad(quad, width, height, min_area):
            quads.append(quad)

    return dedupe_quads(quads)


def hough_line_debug(
    edges: np.ndarray,
    width: int,
    height: int,
    min_area: float,
    variant_idx: int,
    closed_edges_used: bool = False,
) -> tuple[list[Candidate], LineDebug]:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min(width, height) * 0.1,
        maxLineGap=20,
    )
    if lines is None:
        return [], LineDebug(variant_idx, None, None, None, [], closed_edges_used)

    lines = lines.reshape(-1, 4).astype(np.float32)
    families = select_angle_families(lines)
    if families is None:
        return [], LineDebug(variant_idx, lines, None, None, [], closed_edges_used)

    family_a, family_b = families
    pair_a = pair_extreme_lines(family_a)
    pair_b = pair_extreme_lines(family_b)
    quads: list[np.ndarray] = []

    for a0, a1 in pair_a:
        for b0, b1 in pair_b:
            pts = [
                intersect_lines(a0, b0),
                intersect_lines(a1, b0),
                intersect_lines(a1, b1),
                intersect_lines(a0, b1),
            ]
            if any(p is None for p in pts):
                continue
            quad = order_corners(np.asarray(pts, dtype=np.float32))
            if is_valid_quad(quad, width, height, min_area):
                quads.append(quad)

    source = "hough"
    if not quads:
        dominant = family_a if len(family_a) >= len(family_b) else family_b
        quads.extend(dominant_family_quads(edges, dominant, width, height, min_area))
        source = "hough_dominant"

    quads = cap_quads_by_area(quads)
    quads = dedupe_quads(quads)
    candidates = [
        Candidate(quad=quad, source=source, variant_idx=variant_idx) for quad in quads
    ]
    return candidates, LineDebug(
        variant_idx, lines, family_a, family_b, quads, closed_edges_used
    )


def find_hough_candidates(
    edges: np.ndarray, width: int, height: int, min_area: float, variant_idx: int
) -> list[Candidate]:
    candidates, _ = hough_line_debug(edges, width, height, min_area, variant_idx)
    return candidates
