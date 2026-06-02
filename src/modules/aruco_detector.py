from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    run_in_thread = True

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
        self._debug_frame_counter = 0
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
            params = self._aruco.DetectorParameters()
        elif hasattr(self._aruco, "DetectorParameters_create"):
            params = self._aruco.DetectorParameters_create()
        else:
            raise RuntimeError("OpenCV ArUco detector parameters API is unavailable.")

        # Configure detection parameters
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 51
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 3

        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0

        params.polygonalApproxAccuracyRate = 0.08
        params.cornerRefinementMethod = self._aruco.CORNER_REFINE_SUBPIX
        params.errorCorrectionRate = 0.8

        return params

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

    def _preprocess_for_detection(
        self, image: np.ndarray, scale: float = 0.15
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Downscale with AREA then upscale with NEAREST for noise reduction
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(small)

        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

        gray = cv2.bilateralFilter(gray, 5, 30, 30)

        gray = cv2.resize(
            gray,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return gray

    def _detect(
        self, image: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], np.ndarray]:
        processed = self._preprocess_for_detection(image)
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(processed)
        else:
            corners, ids, rejected = self._aruco.detectMarkers(
                processed,
                self.dictionary,
                parameters=self.parameters,
            )
        return list(corners), ids, list(rejected), processed

    def _unpack_detection_result(
        self,
        detection_result: tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]
        | tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], np.ndarray],
        image: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], np.ndarray]:
        if len(detection_result) == 4:
            return detection_result
        if len(detection_result) == 3:
            corners, ids, rejected = detection_result
            return corners, ids, rejected, self._preprocess_for_detection(image)
        raise ValueError("Detector must return 3 or 4 values.")

    @staticmethod
    def _marker_ids(ids: np.ndarray | None) -> list[int]:
        if ids is None or len(ids) == 0:
            return []
        return [int(value) for value in ids.reshape(-1)]

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
            (
                marker_id,
                self._corners_array(marker_corners, float(self.input_border_pixels)),
            )
            for marker_id, marker_corners in zip(
                self._marker_ids(padded_ids), padded_corners
            )
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

    def _draw_rejected(
        self, image: np.ndarray, rejected: list[np.ndarray]
    ) -> np.ndarray:
        output = image.copy()
        if rejected:
            rejected_markers = [
                np.asarray(candidate, dtype=np.float32).reshape(1, 4, 2)
                for candidate in rejected
            ]
            self._aruco.drawDetectedMarkers(
                output, rejected_markers, None, (255, 0, 255)
            )
        return output

    @staticmethod
    def _shift_marker_groups(
        marker_groups: list[np.ndarray],
        offset: float,
    ) -> list[np.ndarray]:
        if offset == 0:
            return [
                np.asarray(group, dtype=np.float32).reshape(1, 4, 2)
                for group in marker_groups
            ]
        return [
            np.asarray(group, dtype=np.float32).reshape(1, 4, 2) + offset
            for group in marker_groups
        ]

    @staticmethod
    def _debug_filename(prefix: str, stem: str, frame_index: int | None = None) -> str:
        if frame_index is None:
            return f"{prefix}{stem}.png"
        return f"{prefix}{stem}_{frame_index:04d}.png"

    def _write_debug_images(
        self,
        image: np.ndarray,
        overlay_image: np.ndarray,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        rejected: list[np.ndarray],
        *,
        prefix: str = "aruco_",
        frame_index: int | None = None,
    ) -> None:
        if not self.debug:
            return
        self._write_debug_image(self._debug_filename(prefix, "input"), image)
        self._write_debug_image(
            self._debug_filename(prefix, "detected_markers", frame_index),
            self._draw_detected(overlay_image, corners, ids),
        )
        self._write_debug_image(
            self._debug_filename(prefix, "rejected_candidates", frame_index),
            self._draw_rejected(overlay_image, rejected),
        )

    async def process(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        return self.process_blocking(message, context)

    def process_blocking(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        payload = message.payload
        frame_index = self._debug_frame_index(payload, message.metadata)
        image = (
            payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        )
        validate_color_image(image)

        raw_corners, raw_ids, raw_rejected, raw_preprocessed = (
            self._unpack_detection_result(self._detect(image), image)
        )
        if self.input_border_pixels > 0:
            padded_image = self._pad_input_image(image, self.input_border_pixels)
            padded_corners, padded_ids, padded_rejected, padded_preprocessed = (
                self._unpack_detection_result(self._detect(padded_image), padded_image)
            )
        else:
            padded_image = image
            padded_corners = []
            padded_ids = None
            padded_rejected = []
            padded_preprocessed = raw_preprocessed

        self._write_debug_image(
            self._debug_filename("aruco_", "preprocessed_raw", frame_index),
            raw_preprocessed,
        )
        self._write_debug_image(
            self._debug_filename("aruco_", "preprocessed_padded", frame_index),
            padded_preprocessed,
        )

        self._warn_on_overlapping_id_mismatches(
            raw_ids,
            raw_corners,
            padded_ids,
            padded_corners,
        )

        detections: list[ArucoMarkerDetection] = []
        detection_passes: dict[int, str] = {}
        seen_marker_ids: set[int] = set()
        debug_union_corners: list[np.ndarray] = []
        raw_debug_corners = self._shift_marker_groups(
            raw_corners, float(self.input_border_pixels)
        )
        padded_debug_corners = self._shift_marker_groups(padded_corners, 0.0)

        for pass_name, detected_corners, detected_ids, offset, debug_corners in (
            ("raw", raw_corners, raw_ids, 0.0, raw_debug_corners),
            (
                "padded",
                padded_corners,
                padded_ids,
                float(self.input_border_pixels),
                padded_debug_corners,
            ),
        ):
            if detected_ids is None or len(detected_ids) == 0:
                continue
            for marker_id, marker_corners, debug_marker_corners in zip(
                self._marker_ids(detected_ids),
                detected_corners,
                debug_corners,
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
                debug_union_corners.append(debug_marker_corners)

        rejected_candidates = [
            np.asarray(candidate, dtype=np.float32).reshape(4, 2)
            for candidate in raw_rejected
        ]
        rejected_candidates.extend(
            np.asarray(candidate, dtype=np.float32).reshape(4, 2)
            - float(self.input_border_pixels)
            for candidate in padded_rejected
        )
        debug_rejected_candidates = [
            candidate.reshape(4, 2)
            for candidate in self._shift_marker_groups(
                raw_rejected,
                float(self.input_border_pixels),
            )
        ]
        debug_rejected_candidates.extend(
            np.asarray(candidate, dtype=np.float32).reshape(4, 2)
            for candidate in padded_rejected
        )

        if not detections:
            self._write_debug_images(
                image,
                padded_image,
                [],
                None,
                debug_rejected_candidates,
                frame_index=frame_index,
            )
            logger.debug("No ArUco markers detected")
            return None

        marker_ids = [detection.marker_id for detection in detections]
        union_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
        self._write_debug_images(
            image,
            padded_image,
            debug_union_corners,
            union_ids,
            debug_rejected_candidates,
            frame_index=frame_index,
        )

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
