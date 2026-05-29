from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import cv2
import numpy as np

from ..marker_rectifier import polygon_area
from .models import _QuadTrackResult

logger = logging.getLogger("src.modules.optical_flow_marker_tracker")


def track_quad_result(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    points: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    max_forward_error: float,
    max_backtrack_error: float,
    min_marker_area: float,
) -> _QuadTrackResult:
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    support_points = quad_support_points(corners)
    previous_points = support_points.reshape(-1, 1, 2)
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_points,
        cast(Any, None),
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None or errors is None:
        return _QuadTrackResult(None, None, "no_forward_flow")

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        next_points,
        cast(Any, None),
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or back_status is None:
        return _QuadTrackResult(None, None, "no_reverse_flow")

    backtrack_errors = np.linalg.norm(
        back_points.reshape(-1, 2) - previous_points.reshape(-1, 2),
        axis=1,
    )
    valid = (
        (status.reshape(-1) == 1)
        & (back_status.reshape(-1) == 1)
        & (errors.reshape(-1) <= max_forward_error)
        & (backtrack_errors <= max_backtrack_error)
    )
    min_valid_points = max(4, int(np.ceil(len(support_points) * 0.25)))
    if int(np.count_nonzero(valid)) < min_valid_points:
        return _QuadTrackResult(None, None, "too_few_valid_points")

    previous_valid = support_points[valid]
    next_valid = next_points.reshape(-1, 2)[valid]
    tracked = transform_corners_from_flow(corners, previous_valid, next_valid)
    if tracked is None:
        return _QuadTrackResult(None, None, "transform_failed")
    tracked = tracked.astype(np.float32)
    if not quad_in_frame(tracked, image_shape):
        return _QuadTrackResult(tracked, None, "out_of_frame")
    if polygon_area(tracked) < min_marker_area:
        return _QuadTrackResult(tracked, None, "below_min_area")

    valid_errors = errors.reshape(-1)[valid]
    valid_backtrack_errors = backtrack_errors[valid]
    valid_ratio = float(np.count_nonzero(valid)) / max(float(len(support_points)), 1.0)
    forward_confidence = 1.0 - min(
        float(np.mean(valid_errors)),
        max_forward_error,
    ) / max(max_forward_error, 1e-6)
    backtrack_confidence = 1.0 - min(
        float(np.mean(valid_backtrack_errors)),
        max_backtrack_error,
    ) / max(max_backtrack_error, 1e-6)
    confidence = max(
        0.0,
        min(1.0, valid_ratio * (forward_confidence + backtrack_confidence) * 0.5),
    )
    return _QuadTrackResult(tracked, confidence, "tracked")


def quad_support_points(corners: np.ndarray, grid_size: int = 5) -> np.ndarray:
    tl, tr, br, bl = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    samples = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    points = []
    for v in samples:
        for u in samples:
            top = (1.0 - u) * tl + u * tr
            bottom = (1.0 - u) * bl + u * br
            points.append((1.0 - v) * top + v * bottom)
    return np.asarray(points, dtype=np.float32)


def transform_corners_from_flow(
    corners: np.ndarray,
    previous_points: np.ndarray,
    next_points: np.ndarray,
) -> np.ndarray | None:
    if len(previous_points) >= 4:
        homography, inliers = cv2.findHomography(
            previous_points.astype(np.float32),
            next_points.astype(np.float32),
            cv2.RANSAC,
            3.0,
        )
        if (
            homography is not None
            and inliers is not None
            and int(np.count_nonzero(inliers)) >= 4
        ):
            transformed = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), homography
            )
            return transformed.reshape(4, 2).astype(np.float32)

    if len(previous_points) >= 3:
        affine, inliers = cv2.estimateAffinePartial2D(
            previous_points.astype(np.float32),
            next_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is not None and (
            inliers is None or int(np.count_nonzero(inliers)) >= 3
        ):
            transformed = cv2.transform(corners.reshape(-1, 1, 2), affine)
            return transformed.reshape(4, 2).astype(np.float32)

    if len(previous_points) >= 1:
        delta = np.median(next_points - previous_points, axis=0).astype(np.float32)
        return (corners + delta).astype(np.float32)
    return None


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(source_points, homography)
    return transformed.reshape(-1, 2).astype(np.float32)


def homography_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray | None:
    raw_homography = metadata.get("cutout_to_source_homography")
    if raw_homography is None:
        logger.warning(
            "Cannot seed optical flow tracking without cutout-to-source homography"
        )
        return None
    homography = np.asarray(raw_homography, dtype=np.float32)
    if homography.shape != (3, 3):
        logger.warning(
            "Cannot seed optical flow tracking with invalid homography shape: %s",
            homography.shape,
        )
        return None
    return homography


def source_quad_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray | None:
    raw_quad = metadata.get("source_quad")
    if raw_quad is None:
        raw_quad = metadata.get("quad")
    if raw_quad is None:
        return None
    quad = np.asarray(raw_quad, dtype=np.float32)
    if quad.shape != (4, 2):
        return None
    return quad


def quad_in_frame(points: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    height, width = image_shape[:2]
    reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return bool(
        np.all(np.isfinite(reshaped))
        and np.all(reshaped[:, 0] >= 0)
        and np.all(reshaped[:, 0] <= width - 1)
        and np.all(reshaped[:, 1] >= 0)
        and np.all(reshaped[:, 1] <= height - 1)
    )
