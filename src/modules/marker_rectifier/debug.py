from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .models import LineDebug

logger = logging.getLogger("src.modules.marker_rectifier")


def draw_line_set(
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


def draw_hough_debug(image: np.ndarray, debug_items: list[LineDebug]) -> np.ndarray:
    canvas = image.copy()
    for item in debug_items:
        raw_color = (120, 120, 120) if not item.closed_edges_used else (80, 80, 160)
        draw_line_set(canvas, item.lines, raw_color, 1)
        draw_line_set(canvas, item.family_a, (255, 0, 0), 2)
        draw_line_set(canvas, item.family_b, (0, 255, 255), 2)
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


def draw_detected_quad(image: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
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


def write_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        logger.warning("Failed to write marker rectifier debug image: %s", path)
