from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import _MarkerTrackDebug, _QuadTrackResult, _TrackedMarker

logger = logging.getLogger("src.modules.optical_flow_marker_tracker")


def optical_flow_debug_path(debug_dir: Path, metadata: dict[str, Any]) -> Path:
    frame_index = metadata.get("frame_index")
    if frame_index is None:
        frame_text = "unknown"
    else:
        frame_text = f"{int(frame_index):05d}"
    return debug_dir / f"frame_{frame_text}_optical_flow_corners.jpg"


def marker_detected_quad_debug_path(debug_dir: Path, metadata: dict[str, Any]) -> Path:
    frame_index = metadata.get("frame_index")
    if frame_index is None:
        frame_text = "unknown"
    else:
        frame_text = f"{int(frame_index):05d}"
    return debug_dir / f"frame_{frame_text}_marker_detected_quad.jpg"


def write_marker_detected_quad_debug_image(
    image: np.ndarray,
    quad: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    debug: bool,
    debug_dir: Path,
) -> None:
    if not debug:
        return

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
    else:
        draw_quad(canvas, quad, (0, 255, 0), label="quad")

    debug_dir.mkdir(parents=True, exist_ok=True)
    path = marker_detected_quad_debug_path(debug_dir, metadata)
    if not cv2.imwrite(str(path), canvas):
        logger.warning("Failed to write tracked-quad debug image: %s", path)


def write_optical_flow_debug_image(
    image: np.ndarray,
    marker_debug: list[_MarkerTrackDebug],
    previous_quad: np.ndarray | None,
    quad_result: _QuadTrackResult | None,
    metadata: dict[str, Any],
    *,
    debug: bool,
    debug_dir: Path,
) -> None:
    if not debug:
        return

    canvas = image.copy()
    draw_optical_flow_debug_summary(canvas, marker_debug, quad_result, metadata)

    if previous_quad is not None:
        if (
            quad_result is not None
            and quad_result.succeeded
            and quad_result.corners is not None
        ):
            draw_quad(canvas, quad_result.corners, (255, 0, 0), label="quad")
        else:
            reason = "missing" if quad_result is None else quad_result.reason
            draw_quad(
                canvas,
                previous_quad,
                (255, 0, 255),
                label=f"quad failed:{reason}",
                dashed=True,
            )
    else:
        cv2.putText(
            canvas,
            "outer quad unavailable",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for item in marker_debug:
        if item.result.succeeded and item.result.corners is not None:
            draw_quad(
                canvas, item.result.corners, (0, 255, 0), label=str(item.marker_id)
            )
            continue
        failed_label = f"{item.marker_id} failed:{item.result.reason}"
        draw_quad(
            canvas,
            item.previous_corners,
            (0, 165, 255),
            label=failed_label,
            dashed=True,
        )
        if item.result.corners is not None:
            draw_quad(
                canvas,
                item.result.corners,
                (0, 0, 255),
                label=f"{item.marker_id} rejected",
            )

    debug_dir.mkdir(parents=True, exist_ok=True)
    path = optical_flow_debug_path(debug_dir, metadata)
    if not cv2.imwrite(str(path), canvas):
        logger.warning("Failed to write optical-flow debug image: %s", path)

    debug_quad = None
    if quad_result is not None and quad_result.corners is not None:
        debug_quad = quad_result.corners
    elif previous_quad is not None:
        debug_quad = previous_quad
    write_marker_detected_quad_debug_image(
        image,
        debug_quad,
        metadata,
        debug=debug,
        debug_dir=debug_dir,
    )


def write_detector_debug_image(
    image: np.ndarray,
    seeded_markers: list[_TrackedMarker],
    source_quad: np.ndarray | None,
    metadata: dict[str, Any],
    *,
    debug: bool,
    debug_dir: Path,
) -> None:
    if not debug:
        return

    canvas = image.copy()
    draw_detector_debug_summary(canvas, seeded_markers, source_quad, metadata)
    if source_quad is not None:
        draw_quad(canvas, source_quad, (255, 0, 0), label="quad detector")
    else:
        cv2.putText(
            canvas,
            "outer quad unavailable",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    for marker in seeded_markers:
        draw_quad(canvas, marker.corners, (0, 255, 255), label=str(marker.marker_id))

    debug_dir.mkdir(parents=True, exist_ok=True)
    path = optical_flow_debug_path(debug_dir, metadata)
    if not cv2.imwrite(str(path), canvas):
        logger.warning("Failed to write detector refresh debug image: %s", path)


def draw_detector_debug_summary(
    image: np.ndarray,
    seeded_markers: list[_TrackedMarker],
    source_quad: np.ndarray | None,
    metadata: dict[str, Any],
) -> None:
    marker_count = len(seeded_markers)
    quad_text = "quad ok" if source_quad is not None else "quad n/a"
    lines = [
        f"detector refresh: markers {marker_count}",
        quad_text,
    ]
    score = metadata.get("score")
    if score is not None:
        lines.append(f"rectifier score {float(score):.3f}")
    _draw_summary_lines(image, lines)


def tracking_failure_summary(
    marker_debug: list[_MarkerTrackDebug],
    quad_result: _QuadTrackResult | None,
) -> str:
    if marker_debug:
        reasons = sorted({item.result.reason for item in marker_debug})
        return ",".join(reasons)
    if quad_result is not None:
        return quad_result.reason
    return "no_trackable_markers"


def draw_optical_flow_debug_summary(
    image: np.ndarray,
    marker_debug: list[_MarkerTrackDebug],
    quad_result: _QuadTrackResult | None,
    metadata: dict[str, Any],
) -> None:
    tracked_count = sum(1 for item in marker_debug if item.result.succeeded)
    attempted_count = len(marker_debug)
    tracking_source = metadata.get("tracking_source", "optical_flow")
    confidence = metadata.get("tracking_confidence")
    confidence_text = (
        "" if confidence is None else f" confidence {float(confidence):.3f}"
    )
    quad_text = "quad n/a"
    if quad_result is not None:
        quad_text = (
            "quad ok" if quad_result.succeeded else f"quad failed:{quad_result.reason}"
        )
    lines = [
        f"{tracking_source}: markers {tracked_count}/{attempted_count}{confidence_text}",
        quad_text,
    ]
    if tracking_source == "optical_flow_failed":
        lines.append(
            f"fallback reason: {metadata.get('tracking_failure_reason', 'unknown')}"
        )
    refresh_reason = metadata.get("tracking_refresh_reason")
    if refresh_reason is not None:
        lines.append(f"refresh: {refresh_reason}")
    _draw_summary_lines(image, lines)


def draw_quad(
    image: np.ndarray,
    quad: np.ndarray,
    color: tuple[int, int, int],
    *,
    label: str,
    dashed: bool = False,
) -> None:
    points = np.rint(np.asarray(quad, dtype=np.float32).reshape(4, 2)).astype(np.int32)
    for idx in range(4):
        p0 = points[idx]
        p1 = points[(idx + 1) % 4]
        if dashed:
            draw_dashed_line(image, p0, p1, color, 2)
        else:
            cv2.line(image, tuple(p0), tuple(p1), color, 2, cv2.LINE_AA)
    for idx, point in enumerate(points):
        point_xy = tuple(int(value) for value in point)
        cv2.circle(image, point_xy, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(idx),
            tuple(int(value) for value in point + np.array([6, -6], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    origin = tuple(
        int(value) for value in points[0] + np.array([8, 18], dtype=np.int32)
    )
    cv2.putText(
        image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )


def draw_dashed_line(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash_length: float = 10.0,
) -> None:
    delta = p1.astype(np.float32) - p0.astype(np.float32)
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    direction = delta / length
    distance = 0.0
    while distance < length:
        start = p0.astype(np.float32) + direction * distance
        end = p0.astype(np.float32) + direction * min(distance + dash_length, length)
        cv2.line(
            image,
            tuple(np.rint(start).astype(np.int32)),
            tuple(np.rint(end).astype(np.int32)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash_length * 2.0


def _draw_summary_lines(image: np.ndarray, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            image,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28
