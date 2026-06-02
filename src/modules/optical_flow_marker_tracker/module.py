from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from ...image import ImageFrame
from ...messages import Message, RoutedMessage
from ...video import VideoFrame
from ..aruco_detector import ArucoDetectionModule, ArucoDetectionResult
from ..base import BaseModule, ModuleContext
from ..image_enhancer import validate_color_image
from ..marker_rectifier import MarkerRectificationModule
from .core import (
    homography_from_metadata,
    quad_in_frame,
    quad_support_points,
    source_quad_from_metadata,
    track_quad_result,
    transform_corners_from_flow,
    transform_points,
)
from .debug import (
    draw_dashed_line,
    draw_detector_debug_summary,
    draw_optical_flow_debug_summary,
    draw_quad,
    marker_detected_quad_debug_path,
    optical_flow_debug_path,
    tracking_failure_summary,
    write_detector_debug_image,
    write_marker_detected_quad_debug_image,
    write_optical_flow_debug_image,
)
from .models import (
    _MarkerTrackDebug,
    _QuadTrackResult,
    _QuadTrackingPlan,
    _TrackedMarker,
    _TrackingState,
)

logger = logging.getLogger("src.modules.optical_flow_marker_tracker")


@dataclass
class _TimingStats:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def record(self, elapsed_seconds: float) -> None:
        self.count += 1
        self.total_seconds += elapsed_seconds
        self.max_seconds = max(self.max_seconds, elapsed_seconds)

    @property
    def average_seconds(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_seconds / self.count


class OpticalFlowMarkerTrackingModule(BaseModule[ImageFrame | VideoFrame | np.ndarray]):
    run_in_thread = True

    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        rectifier: MarkerRectificationModule | None = None,
        detector: ArucoDetectionModule | None = None,
        max_forward_error: float = 25.0,
        max_backtrack_error: float = 3.0,
        min_marker_area: float = 16.0,
        min_tracking_confidence: float = 0.5,
        dictionary_name: str = "DICT_6X6_1000",
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
        emit_empty_detections: bool = False,
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if max_forward_error < 0:
            raise ValueError("max_forward_error cannot be negative.")
        if max_backtrack_error < 0:
            raise ValueError("max_backtrack_error cannot be negative.")
        if min_marker_area < 0:
            raise ValueError("min_marker_area cannot be negative.")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.max_forward_error = max_forward_error
        self.max_backtrack_error = max_backtrack_error
        self.min_marker_area = min_marker_area
        self.min_tracking_confidence = min_tracking_confidence
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.emit_empty_detections = emit_empty_detections
        self.rectifier = rectifier or MarkerRectificationModule(
            name=f"{name}-rectifier",
            input_queue=input_queue,
            output_queue=f"{name}-cutouts",
            debug=debug,
            debug_dir=debug_dir,
        )
        self.detector = detector or ArucoDetectionModule(
            name=f"{name}-aruco-detector",
            input_queue=f"{name}-cutouts",
            dictionary_name=dictionary_name,
            output_queue=output_queue,
            debug=debug,
            debug_dir=debug_dir,
        )
        self._state: _TrackingState | None = None
        self._timing_stats: dict[str, _TimingStats] = {
            stage_name: _TimingStats()
            for stage_name in (
                "total",
                "gray",
                "track_state",
                "rectifier",
                "detector",
                "seed_state",
                "debug_optical_flow",
                "debug_detector",
            )
        }
        self._refresh_reason_counts: dict[str, int] = {}
        self._rectifier_mode_counts: dict[str, int] = {}
        self._timing_logged = False

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
        total_started_at = time.perf_counter()
        payload = message.payload
        image = (
            payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        )
        validate_color_image(image)

        gray_started_at = time.perf_counter()
        gray = self._gray(image)
        self._timing_stats["gray"].record(time.perf_counter() - gray_started_at)

        track_state_started_at = time.perf_counter()
        tracking_plan = self._track_from_state(message, image, gray)
        self._timing_stats["track_state"].record(
            time.perf_counter() - track_state_started_at
        )
        self._increment_count(self._refresh_reason_counts, tracking_plan.refresh_reason)
        self._increment_count(
            self._rectifier_mode_counts,
            "full_search" if tracking_plan.force_full_rectifier else "trusted_prior",
        )

        result = self._run_detector_fallback(
            message,
            context,
            image,
            gray,
            tracking_plan=tracking_plan,
        )
        self._timing_stats["total"].record(time.perf_counter() - total_started_at)
        return result

    def _track_from_state(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        image: np.ndarray,
        gray: np.ndarray,
    ) -> _QuadTrackingPlan:
        state = self._state
        if state is None or state.quad is None:
            return _QuadTrackingPlan(
                prior_quad=None,
                quad_result=None,
                refresh_reason="bootstrap",
                force_full_rectifier=True,
                confidence=None,
            )

        quad_result = self._track_quad_result(state.gray, gray, state.quad, image.shape)
        tracked_quad = quad_result.corners if quad_result.succeeded else None
        quad_confidence = quad_result.confidence if quad_result.succeeded else None
        if tracked_quad is not None and quad_confidence is not None:
            refresh_reason = (
                "trusted_quad"
                if quad_confidence >= self.min_tracking_confidence
                else "low_confidence"
            )
            force_full_rectifier = quad_confidence < self.min_tracking_confidence
            prior_quad = tracked_quad
        else:
            refresh_reason = "quad_track_failed"
            force_full_rectifier = True
            prior_quad = state.quad

        metadata = self._base_metadata(message, image)
        metadata.update(
            {
                "tracking_source": (
                    "optical_flow"
                    if not force_full_rectifier
                    else "optical_flow_failed"
                ),
                "tracking_refresh_reason": refresh_reason,
                "tracked_marker_count": 0,
                "attempted_marker_count": 0,
            }
        )
        if prior_quad is not None:
            metadata["source_quad"] = prior_quad.tolist()
            metadata["quad"] = prior_quad.tolist()
        if quad_confidence is not None:
            metadata["tracking_confidence"] = float(quad_confidence)
            metadata["tracked_quad_confidence"] = float(quad_confidence)
        if not quad_result.succeeded:
            metadata["tracked_quad_failure_reason"] = quad_result.reason

        debug_started_at = time.perf_counter()
        self._write_optical_flow_debug_image(
            image, [], state.quad, quad_result, metadata
        )
        self._timing_stats["debug_optical_flow"].record(
            time.perf_counter() - debug_started_at
        )

        return _QuadTrackingPlan(
            prior_quad=prior_quad,
            quad_result=quad_result,
            refresh_reason=refresh_reason,
            force_full_rectifier=force_full_rectifier,
            confidence=quad_confidence,
        )

    def _run_detector_fallback(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
        image: np.ndarray,
        gray: np.ndarray,
        *,
        tracking_plan: _QuadTrackingPlan,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        fallback_metadata = dict(message.metadata)
        if tracking_plan.prior_quad is not None:
            fallback_metadata["prior_source_quad"] = tracking_plan.prior_quad.tolist()
        if tracking_plan.force_full_rectifier:
            fallback_metadata["force_full_rectifier"] = True
        fallback_metadata["tracking_refresh_reason"] = tracking_plan.refresh_reason
        if tracking_plan.confidence is not None:
            fallback_metadata["tracking_confidence"] = float(tracking_plan.confidence)
            fallback_metadata["tracked_quad_confidence"] = float(
                tracking_plan.confidence
            )
        if (
            tracking_plan.quad_result is not None
            and not tracking_plan.quad_result.succeeded
        ):
            fallback_metadata["tracked_quad_failure_reason"] = (
                tracking_plan.quad_result.reason
            )
        rectifier_started_at = time.perf_counter()
        rectified = self.rectifier.process_blocking(
            Message(payload=message.payload, metadata=fallback_metadata),
            context,
        )
        self._timing_stats["rectifier"].record(
            time.perf_counter() - rectifier_started_at
        )
        if rectified is None:
            self._state = None
            return self._empty_detection_result(message, image)

        detector_started_at = time.perf_counter()
        detected = self.detector.process_blocking(
            cast(Message[ImageFrame | VideoFrame | np.ndarray], rectified.message),
            context,
        )
        self._timing_stats["detector"].record(time.perf_counter() - detector_started_at)
        if detected is None:
            self._state = None
            return self._empty_detection_result(message, image)

        metadata: dict[str, Any] = dict(detected.message.metadata)
        metadata.update(self._base_metadata(message, image))
        metadata["tracking_source"] = "detector"
        metadata.setdefault("detection_coordinate_space", "cutout")
        seed_started_at = time.perf_counter()
        seeded_markers = self._seed_state(gray, detected.message.payload, metadata)
        self._timing_stats["seed_state"].record(time.perf_counter() - seed_started_at)
        debug_started_at = time.perf_counter()
        self._write_detector_debug_image(
            image,
            seeded_markers,
            self._state.quad if self._state is not None else None,
            metadata,
        )
        self._timing_stats["debug_detector"].record(
            time.perf_counter() - debug_started_at
        )

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=detected.message.payload, metadata=metadata),
        )

    def _optical_flow_debug_path(self, metadata: dict[str, Any]) -> Path:
        return optical_flow_debug_path(self.debug_dir, metadata)

    def _marker_detected_quad_debug_path(self, metadata: dict[str, Any]) -> Path:
        return marker_detected_quad_debug_path(self.debug_dir, metadata)

    def _write_marker_detected_quad_debug_image(
        self,
        image: np.ndarray,
        quad: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> None:
        write_marker_detected_quad_debug_image(
            image,
            quad,
            metadata,
            debug=self.debug,
            debug_dir=self.debug_dir,
        )

    def _write_optical_flow_debug_image(
        self,
        image: np.ndarray,
        marker_debug: list[_MarkerTrackDebug],
        previous_quad: np.ndarray | None,
        quad_result: _QuadTrackResult | None,
        metadata: dict[str, Any],
    ) -> None:
        write_optical_flow_debug_image(
            image,
            marker_debug,
            previous_quad,
            quad_result,
            metadata,
            debug=self.debug,
            debug_dir=self.debug_dir,
        )

    def _write_detector_debug_image(
        self,
        image: np.ndarray,
        seeded_markers: list[_TrackedMarker],
        source_quad: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> None:
        write_detector_debug_image(
            image,
            seeded_markers,
            source_quad,
            metadata,
            debug=self.debug,
            debug_dir=self.debug_dir,
        )

    @staticmethod
    def _draw_detector_debug_summary(
        image: np.ndarray,
        seeded_markers: list[_TrackedMarker],
        source_quad: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> None:
        draw_detector_debug_summary(image, seeded_markers, source_quad, metadata)

    @staticmethod
    def _tracking_failure_summary(
        marker_debug: list[_MarkerTrackDebug],
        quad_result: _QuadTrackResult | None,
    ) -> str:
        return tracking_failure_summary(marker_debug, quad_result)

    @staticmethod
    def _draw_optical_flow_debug_summary(
        image: np.ndarray,
        marker_debug: list[_MarkerTrackDebug],
        quad_result: _QuadTrackResult | None,
        metadata: dict[str, Any],
    ) -> None:
        draw_optical_flow_debug_summary(image, marker_debug, quad_result, metadata)

    @staticmethod
    def _draw_quad(
        image: np.ndarray,
        quad: np.ndarray,
        color: tuple[int, int, int],
        *,
        label: str,
        dashed: bool = False,
    ) -> None:
        draw_quad(image, quad, color, label=label, dashed=dashed)

    @staticmethod
    def _draw_dashed_line(
        image: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        *,
        dash_length: float = 10.0,
    ) -> None:
        draw_dashed_line(
            image,
            p0,
            p1,
            color,
            thickness,
            dash_length=dash_length,
        )

    def _empty_detection_result(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        image: np.ndarray,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        if not self.emit_empty_detections:
            return None

        metadata = self._base_metadata(message, image)
        metadata.update(
            {
                "tracking_source": "none",
                "detection_coordinate_space": "source_frame",
                "dictionary_name": None,
                "marker_count": 0,
                "marker_ids": [],
                "detection_passes": {},
                "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                "source_frame_image": image.copy(),
            }
        )
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(
                payload=ArucoDetectionResult(
                    image=image,
                    detections=[],
                    rejected_candidates=[],
                ),
                metadata=metadata,
            ),
        )

    def _seed_state(
        self,
        gray: np.ndarray,
        result: ArucoDetectionResult,
        metadata: dict[str, Any],
    ) -> list[_TrackedMarker]:
        homography = self._homography(metadata)
        if homography is None:
            self._state = None
            return []

        detections = [
            _TrackedMarker(
                marker_id=detection.marker_id,
                corners=self._transform_points(detection.corners, homography),
            )
            for detection in result.detections
        ]
        source_quad = self._source_quad(metadata)
        self._state = _TrackingState(
            gray=gray,
            quad=source_quad,
            dictionary_name=metadata.get("dictionary_name"),
        )
        return detections

    def _track_quad(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        points: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, float] | tuple[None, None]:
        result = self._track_quad_result(previous_gray, gray, points, image_shape)
        if not result.succeeded or result.corners is None or result.confidence is None:
            return None, None
        return result.corners, result.confidence

    def _track_quad_result(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        points: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> _QuadTrackResult:
        return track_quad_result(
            previous_gray,
            gray,
            points,
            image_shape,
            max_forward_error=self.max_forward_error,
            max_backtrack_error=self.max_backtrack_error,
            min_marker_area=self.min_marker_area,
        )

    @staticmethod
    def _quad_support_points(corners: np.ndarray, grid_size: int = 5) -> np.ndarray:
        return quad_support_points(corners, grid_size)

    @staticmethod
    def _transform_corners_from_flow(
        corners: np.ndarray,
        previous_points: np.ndarray,
        next_points: np.ndarray,
    ) -> np.ndarray | None:
        return transform_corners_from_flow(corners, previous_points, next_points)

    def _base_metadata(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        image: np.ndarray,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = dict(message.metadata)
        payload = message.payload
        if isinstance(payload, (ImageFrame, VideoFrame)):
            metadata.update(
                {
                    "frame_index": payload.frame_index,
                    "timestamp_seconds": payload.timestamp_seconds,
                    "loop_count": payload.loop_count,
                }
            )
        metadata["input_shape"] = tuple(int(value) for value in image.shape)
        return metadata

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _increment_count(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def close(self) -> None:
        if self._timing_logged:
            return
        self._timing_logged = True

        total_stats = self._timing_stats["total"]
        if total_stats.count == 0:
            return

        logger.info("Tracker timing summary:")
        for stage_name in (
            "total",
            "gray",
            "track_state",
            "rectifier",
            "detector",
            "seed_state",
            "debug_optical_flow",
            "debug_detector",
        ):
            stats = self._timing_stats[stage_name]
            if stats.count == 0:
                continue
            logger.info(
                "  %s: count=%s avg=%.2fms max=%.2fms total=%.2fms",
                stage_name,
                stats.count,
                stats.average_seconds * 1000.0,
                stats.max_seconds * 1000.0,
                stats.total_seconds * 1000.0,
            )

        logger.info(
            "Tracker refresh reasons: %s",
            ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(self._refresh_reason_counts.items())
            ),
        )
        logger.info(
            "Tracker rectifier modes: %s",
            ", ".join(
                f"{mode}={count}"
                for mode, count in sorted(self._rectifier_mode_counts.items())
            ),
        )

    @staticmethod
    def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        return transform_points(points, homography)

    @staticmethod
    def _homography(metadata: dict[str, Any]) -> np.ndarray | None:
        return homography_from_metadata(metadata)

    @staticmethod
    def _source_quad(metadata: dict[str, Any]) -> np.ndarray | None:
        return source_quad_from_metadata(metadata)

    @staticmethod
    def _quad_in_frame(points: np.ndarray, image_shape: tuple[int, ...]) -> bool:
        return quad_in_frame(points, image_shape)
