from __future__ import annotations

import numpy as np

from .contours import compute_bw_dominance, find_contour_candidates
from .edge_detection import fallback_edge_retry
from .evidence import (
    dedupe_candidate_pool,
    marker_detection_evidence,
    sample_candidates_by_area,
    source_marker_board_quad,
)
from .fitting import candidate_selection_score, edge_distance_score, refine_candidate
from .geometry import is_valid_quad, polygon_area, quad_iou
from .hough import hough_line_debug
from .models import Candidate, EdgeArtifacts, FitResult, LineDebug, MarkerEvidence
from .constants import REFINEMENT_CANDIDATE_CAP


STRONG_MARKER_DETECTION_COUNT = 4
PRIOR_QUAD_IOU_THRESHOLD = 0.55
PRIOR_QUAD_AREA_RATIO_RANGE = (0.55, 1.60)


def _prior_quad_match_score(quad: np.ndarray, prior_quad: np.ndarray | None) -> float:
    if prior_quad is None:
        return 0.0

    quad_area = polygon_area(quad)
    prior_area = polygon_area(prior_quad)
    if quad_area <= 0.0 or prior_area <= 0.0:
        return 0.0

    area_ratio = quad_area / prior_area
    min_ratio, max_ratio = PRIOR_QUAD_AREA_RATIO_RANGE
    if area_ratio < min_ratio or area_ratio > max_ratio:
        return 0.0

    overlap = quad_iou(quad, prior_quad)
    if overlap < PRIOR_QUAD_IOU_THRESHOLD:
        return 0.0
    return float(overlap)


def _candidate_rank(
    marker_evidence: MarkerEvidence,
    selection_score: float,
    prior_match_score: float,
) -> tuple[float, float, float, float, float]:
    if marker_evidence.detected_count >= STRONG_MARKER_DETECTION_COUNT:
        return (
            1.0,
            float(marker_evidence.detected_count),
            float(-marker_evidence.rejected_count),
            float(-selection_score),
            prior_match_score,
        )

    return (
        0.0,
        prior_match_score,
        float(marker_evidence.detected_count),
        float(-marker_evidence.rejected_count),
        float(-selection_score),
    )


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


def _candidate_pre_score(
    image: np.ndarray,
    candidate: Candidate,
    edge_variants: list[EdgeArtifacts],
    width: int,
    height: int,
) -> float:
    artifacts = edge_variants[candidate.variant_idx]
    quad_score = candidate.score
    if quad_score is None:
        quad_score = edge_distance_score(
            candidate.quad, artifacts.dist, artifacts.grad_mag
        )

    quad_bw_dominance = candidate.bw_dominance
    if quad_bw_dominance is None:
        quad_bw_dominance = compute_bw_dominance(image, candidate.quad)

    return candidate_selection_score(
        candidate,
        quad_score,
        width,
        height,
        quad_bw_dominance,
    )


def shortlist_refinement_candidates(
    image: np.ndarray,
    evaluation_pool: list[Candidate],
    edge_variants: list[EdgeArtifacts],
    width: int,
    height: int,
    cap: int = REFINEMENT_CANDIDATE_CAP,
) -> list[Candidate]:
    if len(evaluation_pool) <= cap:
        return evaluation_pool

    selected: list[Candidate] = []
    seen_candidate_ids: set[int] = set()

    for candidate in evaluation_pool:
        if candidate.source not in {"prior", "aruco_board"}:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        selected.append(candidate)

    ranked_candidates = sorted(
        evaluation_pool,
        key=lambda candidate: _candidate_pre_score(
            image,
            candidate,
            edge_variants,
            width,
            height,
        ),
    )
    for candidate in ranked_candidates:
        if len(selected) >= cap:
            break
        candidate_id = id(candidate)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        selected.append(candidate)

    return selected


def fit_square(
    image: np.ndarray,
    edge_variants: list[EdgeArtifacts],
    marker_image: np.ndarray | None = None,
    prior_quad: np.ndarray | None = None,
) -> FitResult:
    height, width = image.shape[:2]
    min_area = max(0.01 * width * height, 400.0)
    all_candidates: list[Candidate] = []
    rejected_debug: list[np.ndarray] = []
    marker_image = image if marker_image is None else marker_image
    if prior_quad is not None:
        prior_quad = np.asarray(prior_quad, dtype=np.float32)
        if prior_quad.shape != (4, 2):
            prior_quad = None

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
    evaluation_pool = list(candidate_pool)
    if prior_quad is None:
        board_quad = source_marker_board_quad(image)
        if board_quad is not None and is_valid_quad(
            board_quad, width, height, min_area
        ):
            evaluation_pool.extend(
                Candidate(quad=board_quad, source="aruco_board", variant_idx=idx)
                for idx in range(len(edge_variants))
            )
    if prior_quad is not None:
        evaluation_pool.extend(
            Candidate(quad=prior_quad, source="prior", variant_idx=idx)
            for idx in range(len(edge_variants))
        )
    evaluation_pool = shortlist_refinement_candidates(
        image,
        evaluation_pool,
        edge_variants,
        width,
        height,
    )

    best_quad = None
    best_score = float("inf")
    best_selection_score = float("inf")
    best_rank: tuple[float, float, float, float, float] | None = None

    for candidate in evaluation_pool:
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
            candidate_rank = _candidate_rank(
                marker_evidence,
                selection_score,
                _prior_quad_match_score(quad, prior_quad),
            )
            is_better = best_rank is None or candidate_rank > best_rank
            if is_better:
                if best_quad is not None:
                    rejected_debug.append(best_quad)
                best_rank = candidate_rank
                best_selection_score = selection_score
                best_score = quad_score
                best_quad = quad
            else:
                rejected_debug.append(quad)

    if best_quad is None:
        raise RuntimeError("Candidate refinement failed")

    return FitResult(quad=best_quad, score=best_score, rejected=rejected_debug)
