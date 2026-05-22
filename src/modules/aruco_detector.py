from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..messages import Message, RoutedMessage
from ..image import ImageFrame
from ..video import VideoFrame
from .base import BaseModule, ModuleContext
from .image_enhancer import validate_color_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArucoMarkerDetection:
    marker_id: int
    corners: np.ndarray


@dataclass(frozen=True)
class ArucoDetectionResult:
    image: np.ndarray
    detections: list[ArucoMarkerDetection]
    rejected_candidates: list[np.ndarray]


class ArucoDetectionModule(BaseModule[ImageFrame | VideoFrame | np.ndarray]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        dictionary_name: str = "DICT_6X6_1000",
        input_border_pixels: int = 16,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if input_border_pixels < 0:
            raise ValueError("input_border_pixels cannot be negative.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.dictionary_name = dictionary_name
        self.input_border_pixels = input_border_pixels
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module.")
        self._aruco = cv2.aruco
        self.dictionary = self._get_dictionary(dictionary_name)
        self.parameters = self._create_detector_parameters()
        self.detector = self._create_detector()

    def _get_dictionary(self, dictionary_name: str) -> Any:
        dictionary_id = getattr(self._aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")
        return self._aruco.getPredefinedDictionary(dictionary_id)

    def _create_detector_parameters(self) -> Any:
        if hasattr(self._aruco, "DetectorParameters"):
            return self._aruco.DetectorParameters()
        if hasattr(self._aruco, "DetectorParameters_create"):
            return self._aruco.DetectorParameters_create()
        raise RuntimeError("OpenCV ArUco detector parameters API is unavailable.")

    def _create_detector(self) -> Any | None:
        if hasattr(self._aruco, "ArucoDetector"):
            return self._aruco.ArucoDetector(self.dictionary, self.parameters)
        return None

    def _pad_input_image(self, image: np.ndarray, border_pixels: int) -> np.ndarray:
        if border_pixels == 0:
            return image
        return cv2.copyMakeBorder(
            image,
            border_pixels,
            border_pixels,
            border_pixels,
            border_pixels,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _detect(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(image)
        else:
            corners, ids, rejected = self._aruco.detectMarkers(
                image,
                self.dictionary,
                parameters=self.parameters,
            )
        return list(corners), ids, list(rejected)

    @staticmethod
    def _marker_ids(ids: np.ndarray | None) -> list[int]:
        if ids is None or len(ids) == 0:
            return []
        return [int(value) for value in ids.reshape(-1)]

    @staticmethod
    def _corners_array(marker_corners: np.ndarray, offset: float = 0.0) -> np.ndarray:
        corners = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        if offset:
            corners = corners - offset
        return corners

    @staticmethod
    def _same_detection_quad(
        first: np.ndarray,
        second: np.ndarray,
        *,
        tolerance_pixels: float = 3.0,
    ) -> bool:
        return bool(
            np.allclose(
                np.asarray(first, dtype=np.float32).reshape(4, 2),
                np.asarray(second, dtype=np.float32).reshape(4, 2),
                atol=tolerance_pixels,
            )
        )

    def _warn_on_overlapping_id_mismatches(
        self,
        raw_ids: np.ndarray | None,
        raw_corners: list[np.ndarray],
        padded_ids: np.ndarray | None,
        padded_corners: list[np.ndarray],
    ) -> None:
        if self.input_border_pixels <= 0:
            return

        raw_detections = [
            (marker_id, self._corners_array(marker_corners))
            for marker_id, marker_corners in zip(self._marker_ids(raw_ids), raw_corners)
        ]
        padded_detections = [
            (marker_id, self._corners_array(marker_corners, float(self.input_border_pixels)))
            for marker_id, marker_corners in zip(self._marker_ids(padded_ids), padded_corners)
        ]

        for raw_id, raw_quad in raw_detections:
            for padded_id, padded_quad in padded_detections:
                if raw_id == padded_id:
                    continue
                if self._same_detection_quad(raw_quad, padded_quad):
                    logger.warning(
                        "ArUco marker id mismatch for same detection quad: raw=%s padded=%s",
                        raw_id,
                        padded_id,
                    )

    def _write_debug_image(self, filename: str, image: np.ndarray) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / filename
        if not cv2.imwrite(str(path), image):
            logger.warning("Failed to write ArUco debug image: %s", path)

    def _draw_detected(
        self,
        image: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> np.ndarray:
        output = image.copy()
        if corners and ids is not None and len(ids) > 0:
            self._aruco.drawDetectedMarkers(output, corners, ids)
        return output

    def _draw_rejected(self, image: np.ndarray, rejected: list[np.ndarray]) -> np.ndarray:
        output = image.copy()
        if rejected:
            rejected_markers = [
                np.asarray(candidate, dtype=np.float32).reshape(1, 4, 2)
                for candidate in rejected
            ]
            self._aruco.drawDetectedMarkers(output, rejected_markers, None, (255, 0, 255))
        return output

    def _write_debug_images(
        self,
        image: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        rejected: list[np.ndarray],
        *,
        prefix: str = "aruco_",
    ) -> None:
        if not self.debug:
            return
        self._write_debug_image(f"{prefix}input.png", image)
        self._write_debug_image(
            f"{prefix}detected_markers.png",
            self._draw_detected(image, corners, ids),
        )
        self._write_debug_image(
            f"{prefix}rejected_candidates.png",
            self._draw_rejected(image, rejected),
        )

    async def process(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        payload = message.payload
        image = payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        validate_color_image(image)

        raw_corners, raw_ids, raw_rejected = self._detect(image)
        if self.input_border_pixels > 0:
            padded_image = self._pad_input_image(image, self.input_border_pixels)
            padded_corners, padded_ids, padded_rejected = self._detect(padded_image)
        else:
            padded_corners = []
            padded_ids = None
            padded_rejected = []

        self._warn_on_overlapping_id_mismatches(
            raw_ids,
            raw_corners,
            padded_ids,
            padded_corners,
        )

        detections: list[ArucoMarkerDetection] = []
        detection_passes: dict[int, str] = {}
        seen_marker_ids: set[int] = set()

        for pass_name, detected_corners, detected_ids, offset in (
            ("raw", raw_corners, raw_ids, 0.0),
            ("padded", padded_corners, padded_ids, float(self.input_border_pixels)),
        ):
            if detected_ids is None or len(detected_ids) == 0:
                continue
            for marker_id, marker_corners in zip(
                self._marker_ids(detected_ids),
                detected_corners,
            ):
                if marker_id in seen_marker_ids:
                    continue
                corners_array = self._corners_array(marker_corners, offset)
                detections.append(
                    ArucoMarkerDetection(
                        marker_id=marker_id,
                        corners=corners_array,
                    )
                )
                detection_passes[marker_id] = pass_name
                seen_marker_ids.add(marker_id)

        rejected_candidates = [
            np.asarray(candidate, dtype=np.float32).reshape(4, 2)
            for candidate in raw_rejected
        ]
        rejected_candidates.extend(
            np.asarray(candidate, dtype=np.float32).reshape(4, 2) - float(self.input_border_pixels)
            for candidate in padded_rejected
        )

        if not detections:
            self._write_debug_images(image, [], None, rejected_candidates)
            logger.debug("No ArUco markers detected")
            return None

        marker_ids = [detection.marker_id for detection in detections]
        union_corners = [
            detection.corners.reshape(1, 4, 2)
            for detection in detections
        ]
        union_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
        self._write_debug_images(image, union_corners, union_ids, rejected_candidates)

        metadata: dict[str, Any] = dict(message.metadata)
        if isinstance(payload, (ImageFrame, VideoFrame)):
            metadata.update(
                {
                    "frame_index": payload.frame_index,
                    "timestamp_seconds": payload.timestamp_seconds,
                    "loop_count": payload.loop_count,
                }
            )
        metadata.update(
            {
                "dictionary_name": self.dictionary_name,
                "marker_count": len(detections),
                "marker_ids": marker_ids,
                "input_border_pixels": self.input_border_pixels,
                "detection_passes": detection_passes,
            }
        )

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(
                payload=ArucoDetectionResult(
                    image=image,
                    detections=detections,
                    rejected_candidates=rejected_candidates,
                ),
                metadata=metadata,
            ),
        )
