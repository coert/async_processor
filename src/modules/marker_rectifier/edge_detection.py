from __future__ import annotations

import cv2
import numpy as np

from ..image_enhancer import EnhancementMode, apply_enhancement
from .models import EdgeArtifacts


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
