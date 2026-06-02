from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ...image import ImageFrame
from ...messages import Message, RoutedMessage
from ...video import VideoFrame
from ..base import BaseModule, ModuleContext
from ..image_enhancer import EnhancementMode, validate_color_image
from .core import collect_line_debug, fit_square
from .debug import draw_detected_quad, draw_hough_debug, write_debug_image
from .edge_detection import build_edge_variants
from .models import LineDebug
from .perspective import perspective_transform_matrices

logger = logging.getLogger("src.modules.marker_rectifier")


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

    def _debug_input_path(self, fidx: int) -> Path:
        return self.debug_dir / f"frame_{fidx:05d}_input.jpg"

    def _debug_hough_lines_path(self, fidx: int) -> Path:
        return self.debug_dir / f"frame_{fidx:05d}_hough_lines.jpg"

    def _debug_detected_quad_path(self, fidx: int) -> Path:
        return self.debug_dir / f"frame_{fidx:05d}_detected_quad.jpg"

    def _debug_cutout_path(self, fidx: int) -> Path:
        return self.debug_dir / f"frame_{fidx:05d}_rectified_cutout.jpg"

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

    @staticmethod
    def _quad_detection_image(
        image: np.ndarray, _metadata: Mapping[str, Any]
    ) -> np.ndarray:
        return image

    @staticmethod
    def _prior_quad(metadata: Mapping[str, Any]) -> np.ndarray | None:
        raw_prior_quad = metadata.get("prior_source_quad")
        if raw_prior_quad is None:
            return None
        prior_quad = np.asarray(raw_prior_quad, dtype=np.float32)
        if prior_quad.shape != (4, 2):
            return None
        return prior_quad

    def _write_debug_images(
        self,
        fidx: int,
        debug_image: np.ndarray,
        line_debug: list[LineDebug],
        quad: np.ndarray | None,
        cutout: np.ndarray | None,
    ) -> None:
        if not self.debug:
            return

        write_debug_image(self._debug_input_path(fidx), debug_image)
        write_debug_image(
            self._debug_hough_lines_path(fidx),
            draw_hough_debug(debug_image, line_debug),
        )
        write_debug_image(
            self._debug_detected_quad_path(fidx),
            draw_detected_quad(debug_image, quad),
        )
        if cutout is None:
            cutout = np.zeros(
                (self.out_size, self.out_size, debug_image.shape[2]),
                dtype=debug_image.dtype,
            )
        write_debug_image(self._debug_cutout_path(fidx), cutout)

    @staticmethod
    def _force_full_rectifier(metadata: Mapping[str, Any]) -> bool:
        return bool(metadata.get("force_full_rectifier", False))

    async def process(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray] | None:
        payload = message.payload
        fidx = self._debug_frame_index(payload, message.metadata)
        logger.debug("Processing frame index %d with marker rectification module", fidx)
        image = (
            payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        )
        validate_color_image(image)
        quad_image = self._quad_detection_image(image, message.metadata)
        prior_quad = self._prior_quad(message.metadata)
        force_full_rectifier = self._force_full_rectifier(message.metadata)

        line_debug: list[LineDebug] = []
        rectified_quad: np.ndarray
        rectifier_score = 0.0
        rectifier_search_mode = "trusted_prior"
        try:
            if prior_quad is None or force_full_rectifier:
                rectifier_search_mode = "full_search"
                _, edge_variants = build_edge_variants(quad_image, self.preprocess_mode)
                if self.debug:
                    line_debug = collect_line_debug(quad_image, edge_variants)
                fit_result = fit_square(
                    quad_image,
                    edge_variants,
                    marker_image=image,
                    prior_quad=prior_quad,
                )
                rectified_quad = fit_result.quad
                rectifier_score = float(fit_result.score)
            else:
                rectified_quad = prior_quad
        except RuntimeError as exc:
            self._write_debug_images(
                fidx, quad_image, line_debug, quad=None, cutout=None
            )
            logger.warning("Dropping frame without detected marker: %s", exc)
            return None

        source_to_cutout, cutout_to_source = perspective_transform_matrices(
            rectified_quad,
            self.out_size,
        )
        cutout = cv2.warpPerspective(
            image, source_to_cutout, (self.out_size, self.out_size)
        )
        self._write_debug_images(
            fidx, quad_image, line_debug, quad=rectified_quad, cutout=cutout
        )
        metadata: dict[str, Any] = dict(message.metadata)
        metadata.update(
            {
                "quad": rectified_quad.tolist(),
                "source_quad": rectified_quad.tolist(),
                "score": rectifier_score,
                "rectifier_search_mode": rectifier_search_mode,
                "input_shape": tuple(int(value) for value in image.shape),
                "source_frame_image": image.copy(),
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
            rectifier_score,
            self.out_size,
            self.out_size,
        )
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=cutout, metadata=metadata),
        )
