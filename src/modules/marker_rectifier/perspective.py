from __future__ import annotations

import cv2
import numpy as np


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
