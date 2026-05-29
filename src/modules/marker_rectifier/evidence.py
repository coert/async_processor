from __future__ import annotations

from typing import Any, Sequence, cast

import cv2
import numpy as np

from .constants import (
    ARUCO_DICT_NAME,
    ARUCO_INPUT_BORDER_PIXELS,
    HOUGH_CANDIDATE_SAMPLE_COUNT,
)
from .geometry import order_corners, polygon_area
from .models import Candidate, MarkerEvidence
from .perspective import warp_square_cutout

_ARUCO_DETECTOR_STATE: tuple[Any, Any, Any, Any] | None = None

KNOWN_BOARD_MARKER_CENTERS: dict[int, np.ndarray] = {
    3: np.array([0.8535505, 0.31165478], dtype=np.float32),
    8: np.array([0.8512896, 0.7691834], dtype=np.float32),
    22: np.array([0.5218096, 0.2975513], dtype=np.float32),
    39: np.array([0.5166197, 0.75771636], dtype=np.float32),
    40: np.array([0.1856637, 0.2838378], dtype=np.float32),
    41: np.array([0.17645948, 0.7498998], dtype=np.float32),
}
_KNOWN_BOARD_CORNERS = np.array(
    [[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]], [[0.0, 1.0]]],
    dtype=np.float32,
)


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
        _, ids, rejected = detector.detectMarkers(padded)
    elif hasattr(aruco, "detectMarkers"):
        _, ids, rejected = aruco.detectMarkers(
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


def source_marker_board_quad(
    image: np.ndarray,
    min_known_markers: int = 4,
) -> np.ndarray | None:
    state = _aruco_detector_state()
    if state is None:
        return None

    aruco, dictionary, parameters, detector = state
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(image)
    elif hasattr(aruco, "detectMarkers"):
        corners, ids, _ = aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters,
        )
    else:
        return None

    if ids is None:
        return None

    source_points: list[np.ndarray] = []
    detected_points: list[np.ndarray] = []
    for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
        canonical_center = KNOWN_BOARD_MARKER_CENTERS.get(int(marker_id))
        if canonical_center is None:
            continue
        detected_center = (
            np.asarray(marker_corners, dtype=np.float32).reshape(4, 2).mean(axis=0)
        )
        source_points.append(canonical_center)
        detected_points.append(detected_center.astype(np.float32))

    if len(source_points) < min_known_markers:
        return None

    homography, inliers = cv2.findHomography(
        np.asarray(source_points, dtype=np.float32),
        np.asarray(detected_points, dtype=np.float32),
        cv2.RANSAC,
        3.0,
    )
    if homography is None:
        return None
    if inliers is not None and int(np.count_nonzero(inliers)) < min_known_markers:
        return None

    quad = cv2.perspectiveTransform(_KNOWN_BOARD_CORNERS, homography).reshape(4, 2)
    return order_corners(quad.astype(np.float32))


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
