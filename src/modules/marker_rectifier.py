from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..messages import Message, RoutedMessage
from ..image import ImageFrame
from ..video import VideoFrame
from .base import BaseModule, ModuleContext
from .image_enhancer import (
    ORIGINAL_FRAME_METADATA_KEY,
    EnhancementMode,
    apply_enhancement,
    validate_color_image,
)

logger = logging.getLogger(__name__)


@dataclass
class EdgeArtifacts:
    gray: np.ndarray
    blur: np.ndarray
    edges_canny: np.ndarray
    grad_mag: np.ndarray
    dist: np.ndarray


def detect_edges(
    gray: np.ndarray, low_scale: float = 0.66, high_scale: float = 1.33
) -> EdgeArtifacts:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(blur))
    low = int(max(0, low_scale * median))
    high = int(min(255, high_scale * median))
    if high <= low:
        high = min(255, low + 32)

    edges_canny = cv2.Canny(blur, low, high)
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    edge_binary = edges_canny > 0
    dist = cv2.distanceTransform((~edge_binary).astype(np.uint8), cv2.DIST_L2, 5)

    return EdgeArtifacts(
        gray=gray,
        blur=blur,
        edges_canny=edges_canny,
        grad_mag=grad_mag,
        dist=dist,
    )


def build_edge_variants(
    image: np.ndarray, preprocess_mode: EnhancementMode = None
) -> tuple[np.ndarray, list[EdgeArtifacts]]:
    working_image = (
        apply_enhancement(image, preprocess_mode)
        if preprocess_mode is not None
        else image
    )
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    variants = [
        detect_edges(gray),
        detect_edges(gray, low_scale=0.5, high_scale=1.1),
    ]
    return working_image, variants


def fallback_edge_retry(artifacts: EdgeArtifacts) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(artifacts.edges_canny, cv2.MORPH_CLOSE, kernel)


EPSILONS = [0.01, 0.02, 0.03, 0.05, 0.08]
BW_BLACK_THRESHOLD = 72
BW_WHITE_THRESHOLD = 180
BW_NEUTRAL_CHROMA_THRESHOLD = 18.0
BW_PRIORITY_THRESHOLD = 0.55
NMS_IOU_THRESHOLD = 0.35
ARUCO_DICT_NAME = "DICT_6X6_1000"
ARUCO_INPUT_BORDER_PIXELS = 16
HOUGH_FAMILY_SAMPLE_COUNT = 14
HOUGH_CANDIDATE_SAMPLE_COUNT = 32

_ARUCO_DETECTOR_STATE: tuple[Any, Any, Any, Any] | None = None


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


def _aruco_detector_state() -> tuple[Any, Any, Any, Any] | None:
    global _ARUCO_DETECTOR_STATE
    if _ARUCO_DETECTOR_STATE is not None:
        return _ARUCO_DETECTOR_STATE
    if not hasattr(cv2, "aruco"):
        return None

    aruco = cv2.aruco
    dictionary_id = getattr(aruco, ARUCO_DICT_NAME, None)
    if dictionary_id is None:
        return None
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters_factory = getattr(aruco, "DetectorParameters_create", None)
        if not callable(parameters_factory):
            return None
        parameters = parameters_factory()
    detector = (
        aruco.ArucoDetector(dictionary, cast(Any, parameters))
        if hasattr(aruco, "ArucoDetector")
        else None
    )
    _ARUCO_DETECTOR_STATE = (aruco, dictionary, parameters, detector)
    return _ARUCO_DETECTOR_STATE


def marker_detection_evidence(
    image: np.ndarray,
    quad: np.ndarray,
    out_size: int = 512,
) -> MarkerEvidence:
    state = _aruco_detector_state()
    if state is None:
        return MarkerEvidence(detected_count=0, rejected_count=0)

    aruco, dictionary, parameters, detector = state
    cutout = warp_square_cutout(image, quad, out_size)
    padded = cv2.copyMakeBorder(
        cutout,
        ARUCO_INPUT_BORDER_PIXELS,
        ARUCO_INPUT_BORDER_PIXELS,
        ARUCO_INPUT_BORDER_PIXELS,
        ARUCO_INPUT_BORDER_PIXELS,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(padded)
    elif hasattr(aruco, "detectMarkers"):
        corners, ids, rejected = aruco.detectMarkers(
            padded,
            dictionary,
            parameters=parameters,
        )
    else:
        return MarkerEvidence(detected_count=0, rejected_count=0)

    return MarkerEvidence(
        detected_count=0 if ids is None else int(len(ids)),
        rejected_count=int(len(rejected)),
    )


def dedupe_candidates(
    quads: Iterable[np.ndarray], tol: float = 10.0
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for quad in quads:
        quad = order_corners(quad)
        if any(np.mean(np.linalg.norm(quad - other, axis=1)) < tol for other in unique):
            continue
        unique.append(quad)
    return unique


def sample_candidates_by_area(
    candidates: Sequence[Candidate], sample_count: int = HOUGH_CANDIDATE_SAMPLE_COUNT
) -> list[Candidate]:
    if len(candidates) <= sample_count:
        return list(candidates)

    ordered = sorted(
        candidates, key=lambda candidate: polygon_area(candidate.quad), reverse=True
    )
    positions = np.linspace(0, len(ordered) - 1, num=sample_count, dtype=int)
    sampled: list[Candidate] = []
    seen_positions: set[int] = set()
    for position in positions:
        position_int = int(position)
        if position_int in seen_positions:
            continue
        seen_positions.add(position_int)
        sampled.append(ordered[position_int])
    return sampled


def dedupe_candidate_pool(
    candidates: Sequence[Candidate], tol: float = 10.0
) -> list[Candidate]:
    unique: list[Candidate] = []
    for candidate in candidates:
        if any(
            np.mean(np.linalg.norm(candidate.quad - other.quad, axis=1)) < tol
            for other in unique
        ):
            continue
        unique.append(candidate)
    return unique


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
        for quad in dedupe_candidates(candidates)
    ]


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
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
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

    return dedupe_candidates(quads)


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

    quads = dedupe_candidates(quads)
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

    # A single dominant line family can lock onto internal stripes. Prefer the
    # larger projected sign without forcing equal side lengths in image space.
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


def collect_line_debug(
    image: np.ndarray, edge_variants: list[EdgeArtifacts]
) -> list[LineDebug]:
    height, width = image.shape[:2]
    min_area = max(0.01 * width * height, 400.0)
    debug_items: list[LineDebug] = []
    for idx, artifacts in enumerate(edge_variants):
        _, debug = hough_line_debug(artifacts.edges_canny, width, height, min_area, idx)
        debug_items.append(debug)
        closed_edges = fallback_edge_retry(artifacts)
        _, closed_debug = hough_line_debug(
            closed_edges, width, height, min_area, idx, closed_edges_used=True
        )
        debug_items.append(closed_debug)
    return debug_items


def fit_square(
    image: np.ndarray,
    edge_variants: list[EdgeArtifacts],
    marker_image: np.ndarray | None = None,
) -> FitResult:
    height, width = image.shape[:2]
    min_area = max(0.01 * width * height, 400.0)
    all_candidates: list[Candidate] = []
    rejected_debug: list[np.ndarray] = []
    marker_image = image if marker_image is None else marker_image

    for idx, artifacts in enumerate(edge_variants):
        contour_candidates = find_contour_candidates(
            artifacts.edges_canny, width, height, min_area, idx
        )
        hough_candidates, _ = hough_line_debug(
            artifacts.edges_canny, width, height, min_area, idx
        )
        closed_edges = fallback_edge_retry(artifacts)
        closed_contour_candidates = find_contour_candidates(
            closed_edges, width, height, min_area, idx
        )
        closed_hough_candidates, _ = hough_line_debug(
            closed_edges, width, height, min_area, idx, closed_edges_used=True
        )
        candidates = (
            contour_candidates
            + sample_candidates_by_area(hough_candidates)
            + closed_contour_candidates
            + sample_candidates_by_area(closed_hough_candidates)
        )

        for candidate in candidates:
            candidate.score = edge_distance_score(
                candidate.quad, artifacts.dist, artifacts.grad_mag
            )
            candidate.bw_dominance = compute_bw_dominance(image, candidate.quad)
            all_candidates.append(candidate)

    if not all_candidates:
        raise RuntimeError("No valid square candidate found")

    candidate_pool = dedupe_candidate_pool(all_candidates)

    best_quad = None
    best_score = float("inf")
    best_selection_score = float("inf")
    best_marker_evidence = MarkerEvidence(detected_count=-1, rejected_count=10**9)

    for candidate in candidate_pool:
        artifacts = edge_variants[candidate.variant_idx]
        reg_weight = 0.20 if candidate.source == "hough_dominant" else 0.05
        refined = refine_candidate(
            candidate.quad,
            artifacts.dist,
            width,
            height,
            min_area,
            reg_weight=reg_weight,
        )
        evaluated_quads = [candidate.quad]
        if not np.allclose(refined, candidate.quad, atol=1.0):
            evaluated_quads.append(refined)

        for quad in evaluated_quads:
            quad_score = edge_distance_score(quad, artifacts.dist, artifacts.grad_mag)
            quad_bw_dominance = compute_bw_dominance(image, quad)
            evaluated_candidate = Candidate(
                quad=quad,
                source=candidate.source,
                variant_idx=candidate.variant_idx,
                bw_dominance=quad_bw_dominance,
            )
            selection_score = candidate_selection_score(
                evaluated_candidate,
                quad_score,
                width,
                height,
                quad_bw_dominance,
            )
            marker_evidence = marker_detection_evidence(marker_image, quad)
            is_better = (
                marker_evidence.detected_count,
                -marker_evidence.rejected_count,
                -selection_score,
            ) > (
                best_marker_evidence.detected_count,
                -best_marker_evidence.rejected_count,
                -best_selection_score,
            )
            if is_better:
                if best_quad is not None:
                    rejected_debug.append(best_quad)
                best_marker_evidence = marker_evidence
                best_selection_score = selection_score
                best_score = quad_score
                best_quad = quad
            else:
                rejected_debug.append(quad)

    if best_quad is None:
        raise RuntimeError("Candidate refinement failed")

    return FitResult(quad=best_quad, score=best_score, rejected=rejected_debug)


def square_cutout_points(out_size: int) -> np.ndarray:
    return np.array(
        [
            [0, 0],
            [out_size - 1, 0],
            [out_size - 1, out_size - 1],
            [0, out_size - 1],
        ],
        dtype=np.float32,
    )


def perspective_transform_matrices(
    quad: np.ndarray,
    out_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dst_square = square_cutout_points(out_size)
    source_quad = quad.astype(np.float32)
    source_to_cutout = cv2.getPerspectiveTransform(source_quad, dst_square)
    cutout_to_source = cv2.getPerspectiveTransform(dst_square, source_quad)
    return source_to_cutout, cutout_to_source


def warp_square_cutout(
    image: np.ndarray, quad: np.ndarray, out_size: int
) -> np.ndarray:
    source_to_cutout, _ = perspective_transform_matrices(quad, out_size)
    return cv2.warpPerspective(image, source_to_cutout, (out_size, out_size))


def _draw_line_set(
    canvas: np.ndarray,
    lines: np.ndarray | None,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if lines is None:
        return
    for x1, y1, x2, y2 in lines.astype(np.int32):
        cv2.line(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_hough_debug(image: np.ndarray, debug_items: list[LineDebug]) -> np.ndarray:
    canvas = image.copy()
    for item in debug_items:
        raw_color = (120, 120, 120) if not item.closed_edges_used else (80, 80, 160)
        _draw_line_set(canvas, item.lines, raw_color, 1)
        _draw_line_set(canvas, item.family_a, (255, 0, 0), 2)
        _draw_line_set(canvas, item.family_b, (0, 255, 255), 2)
        for quad in item.candidates:
            cv2.polylines(
                canvas,
                [np.rint(quad).astype(np.int32)],
                True,
                (0, 180, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _draw_detected_quad(image: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    canvas = image.copy()
    if quad is None:
        cv2.putText(
            canvas,
            "NO MARKER DETECTED",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        return canvas

    points = np.rint(quad).astype(np.int32)
    cv2.polylines(canvas, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(
            canvas,
            tuple(int(value) for value in point),
            7,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(idx),
            tuple(int(value) for value in point + np.array([8, -8], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _write_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        logger.warning("Failed to write marker rectifier debug image: %s", path)


class MarkerRectificationModule(BaseModule[ImageFrame | VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        out_size: int = 512,
        preprocess_mode: EnhancementMode = None,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if out_size <= 0:
            raise ValueError("out_size must be greater than zero.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.out_size = out_size
        self.preprocess_mode: EnhancementMode = preprocess_mode
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self._debug_frame_counter = 0
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _debug_input_path(self) -> Path:
        return self.debug_dir / "marker_input.png"

    @property
    def _debug_hough_lines_path(self) -> Path:
        return self.debug_dir / "marker_hough_lines.png"

    def _debug_detected_quad_path(self, fidx: int) -> Path:
        return self.debug_dir / f"marker_detected_quad_{fidx:04}.png"

    def _debug_frame_index(
        self, payload: ImageFrame | VideoFrame | np.ndarray, metadata: Mapping[str, Any]
    ) -> int:
        frame_index: int | None = None
        if isinstance(payload, (ImageFrame, VideoFrame)):
            frame_index = int(payload.frame_index)
        else:
            raw_frame_index = metadata.get("frame_index")
            if isinstance(raw_frame_index, (int, np.integer)):
                frame_index = int(raw_frame_index)

        if frame_index is not None:
            self._debug_frame_counter = max(self._debug_frame_counter, frame_index + 1)
            return frame_index

        frame_index = self._debug_frame_counter
        self._debug_frame_counter += 1
        return frame_index

    @property
    def _debug_cutout_path(self) -> Path:
        return self.debug_dir / "marker_rectified_cutout.png"

    def _quad_detection_image(
        self,
        image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> np.ndarray:
        original_frame = metadata.get(ORIGINAL_FRAME_METADATA_KEY)
        if not isinstance(original_frame, np.ndarray):
            return image
        if original_frame.shape != image.shape or original_frame.dtype != image.dtype:
            return image
        try:
            validate_color_image(original_frame)
        except TypeError, ValueError:
            return image
        return original_frame

    def _write_debug_images(
        self,
        fidx: int,
        image: np.ndarray,
        line_debug: list[LineDebug],
        quad: np.ndarray | None,
        cutout: np.ndarray | None,
    ) -> None:
        if not self.debug:
            return

        _write_debug_image(self._debug_input_path, image)
        _write_debug_image(
            self._debug_hough_lines_path, _draw_hough_debug(image, line_debug)
        )
        _write_debug_image(
            self._debug_detected_quad_path(fidx), _draw_detected_quad(image, quad)
        )
        if cutout is None:
            cutout = np.zeros(
                (self.out_size, self.out_size, image.shape[2]), dtype=image.dtype
            )
        _write_debug_image(self._debug_cutout_path, cutout)

    async def process(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray] | None:
        payload = message.payload
        fidx = self._debug_frame_index(payload, message.metadata)
        image = (
            payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        )
        validate_color_image(image)
        quad_image = self._quad_detection_image(image, message.metadata)

        line_debug: list[LineDebug] = []
        try:
            _, edge_variants = build_edge_variants(quad_image, self.preprocess_mode)
            if self.debug:
                line_debug = collect_line_debug(quad_image, edge_variants)
            fit_result = fit_square(quad_image, edge_variants, marker_image=image)
        except RuntimeError as exc:
            self._write_debug_images(
                fidx, quad_image, line_debug, quad=None, cutout=None
            )
            logger.warning("Dropping frame without detected marker: %s", exc)
            return None

        source_to_cutout, cutout_to_source = perspective_transform_matrices(
            fit_result.quad,
            self.out_size,
        )
        cutout = cv2.warpPerspective(
            image, source_to_cutout, (self.out_size, self.out_size)
        )
        self._write_debug_images(
            fidx, quad_image, line_debug, quad=fit_result.quad, cutout=cutout
        )
        metadata: dict[str, Any] = dict(message.metadata)
        metadata.update(
            {
                "quad": fit_result.quad.tolist(),
                "source_quad": fit_result.quad.tolist(),
                "score": float(fit_result.score),
                "input_shape": tuple(int(value) for value in image.shape),
                "source_frame_image": quad_image.copy(),
                "cutout_size": int(self.out_size),
                "source_to_cutout_homography": source_to_cutout.tolist(),
                "cutout_to_source_homography": cutout_to_source.tolist(),
            }
        )
        if isinstance(payload, (ImageFrame, VideoFrame)):
            metadata.update(
                {
                    "frame_index": payload.frame_index,
                    "timestamp_seconds": payload.timestamp_seconds,
                    "loop_count": payload.loop_count,
                }
            )

        logger.debug(
            "Rectified marker with score %.3f to %sx%s cutout",
            fit_result.score,
            self.out_size,
            self.out_size,
        )
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=cutout, metadata=metadata),
        )
