from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..image import ImageFrame
from ..messages import Message, RoutedMessage
from ..video import VideoFrame
from .aruco_detector import ArucoDetectionModule, ArucoDetectionResult, ArucoMarkerDetection
from .base import BaseModule, ModuleContext
from .image_enhancer import validate_color_image
from .marker_rectifier import MarkerRectificationModule, polygon_area

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrackedMarker:
    marker_id: int
    corners: np.ndarray


@dataclass(frozen=True)
class _TrackingState:
    gray: np.ndarray
    detections: list[_TrackedMarker]
    quad: np.ndarray | None
    dictionary_name: str | None


@dataclass(frozen=True)
class _QuadTrackResult:
    corners: np.ndarray | None
    confidence: float | None
    reason: str

    @property
    def succeeded(self) -> bool:
        return self.corners is not None and self.confidence is not None


@dataclass(frozen=True)
class _MarkerTrackDebug:
    marker_id: int
    previous_corners: np.ndarray
    result: _QuadTrackResult


class OpticalFlowMarkerTrackingModule(BaseModule[ImageFrame | VideoFrame | np.ndarray]):
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

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.max_forward_error = max_forward_error
        self.max_backtrack_error = max_backtrack_error
        self.min_marker_area = min_marker_area
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
            output_queue=output_queue,
            debug=debug,
            debug_dir=debug_dir,
        )
        self._state: _TrackingState | None = None

    async def process(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        payload = message.payload
        image = payload.image if isinstance(payload, (ImageFrame, VideoFrame)) else payload
        validate_color_image(image)

        gray = self._gray(image)
        tracked = self._track_from_state(message, image, gray)
        if tracked is not None:
            return tracked

        return await self._run_detector_fallback(message, context, image, gray)

    def _track_from_state(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        image: np.ndarray,
        gray: np.ndarray,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        state = self._state
        if state is None or not state.detections:
            return None

        marker_debug: list[_MarkerTrackDebug] = []
        tracked_markers: list[_TrackedMarker] = []
        marker_confidences: list[float] = []
        for detection in state.detections:
            result = self._track_quad_result(state.gray, gray, detection.corners, image.shape)
            marker_debug.append(
                _MarkerTrackDebug(
                    marker_id=detection.marker_id,
                    previous_corners=detection.corners,
                    result=result,
                )
            )
            if not result.succeeded or result.corners is None or result.confidence is None:
                continue
            tracked_markers.append(_TrackedMarker(detection.marker_id, result.corners))
            marker_confidences.append(result.confidence)

        quad_result: _QuadTrackResult | None = None
        if state.quad is not None:
            quad_result = self._track_quad_result(state.gray, gray, state.quad, image.shape)
        tracked_quad = quad_result.corners if quad_result is not None and quad_result.succeeded else None
        quad_confidence = quad_result.confidence if quad_result is not None and quad_result.succeeded else None

        metadata = self._base_metadata(message, image)
        if not tracked_markers:
            metadata.update(
                {
                    "tracking_source": "optical_flow_failed",
                    "tracking_failure_reason": self._tracking_failure_summary(marker_debug, quad_result),
                    "tracked_marker_count": 0,
                    "attempted_marker_count": len(marker_debug),
                }
            )
            self._write_optical_flow_debug_image(
                image,
                marker_debug,
                state.quad,
                quad_result,
                metadata,
            )
            self._state = None
            return None

        self._state = _TrackingState(
            gray=gray,
            detections=tracked_markers,
            quad=tracked_quad,
            dictionary_name=state.dictionary_name,
        )

        detections = [
            ArucoMarkerDetection(marker_id=marker.marker_id, corners=marker.corners)
            for marker in tracked_markers
        ]
        marker_ids = [detection.marker_id for detection in detections]
        confidence = float(np.mean(marker_confidences)) if marker_confidences else 0.0

        metadata.update(
            {
                "tracking_source": "optical_flow",
                "tracking_confidence": confidence,
                "tracked_marker_count": len(detections),
                "attempted_marker_count": len(marker_debug),
                "detection_coordinate_space": "source_frame",
                "dictionary_name": state.dictionary_name,
                "marker_count": len(detections),
                "marker_ids": marker_ids,
                "detection_passes": {marker_id: "optical_flow" for marker_id in marker_ids},
                "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                "source_frame_image": image.copy(),
            }
        )
        if tracked_quad is not None:
            metadata["source_quad"] = tracked_quad.tolist()
            metadata["quad"] = tracked_quad.tolist()
        if quad_confidence is not None:
            metadata["tracked_quad_confidence"] = float(quad_confidence)
        if quad_result is not None and not quad_result.succeeded:
            metadata["tracked_quad_failure_reason"] = quad_result.reason

        self._write_optical_flow_debug_image(
            image,
            marker_debug,
            state.quad,
            quad_result,
            metadata,
        )

        logger.debug(
            "Tracked %s/%s marker(s) with optical flow confidence %.3f",
            len(detections),
            len(marker_debug),
            confidence,
        )
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(
                payload=ArucoDetectionResult(
                    image=image,
                    detections=detections,
                    rejected_candidates=[],
                ),
                metadata=metadata,
            ),
        )

    async def _run_detector_fallback(
        self,
        message: Message[ImageFrame | VideoFrame | np.ndarray],
        context: ModuleContext,
        image: np.ndarray,
        gray: np.ndarray,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        rectified = await self.rectifier.process(message, context)
        if rectified is None:
            self._state = None
            return self._empty_detection_result(message, image)

        detected = await self.detector.process(rectified.message, context)
        if detected is None:
            self._state = None
            return self._empty_detection_result(message, image)

        metadata: dict[str, Any] = dict(detected.message.metadata)
        metadata.update(self._base_metadata(message, image))
        metadata["tracking_source"] = "detector"
        metadata.setdefault("detection_coordinate_space", "cutout")
        seeded_markers = self._seed_state(gray, detected.message.payload, metadata)
        self._write_detector_debug_image(image, seeded_markers, self._state.quad if self._state is not None else None, metadata)

        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=detected.message.payload, metadata=metadata),
        )

    def _optical_flow_debug_path(self, metadata: dict[str, Any]) -> Path:
        frame_index = metadata.get("frame_index")
        if frame_index is None:
            frame_text = "unknown"
        else:
            frame_text = f"{int(frame_index):06d}"
        return self.debug_dir / f"optical_flow_corners_frame_{frame_text}.png"

    def _write_optical_flow_debug_image(
        self,
        image: np.ndarray,
        marker_debug: list[_MarkerTrackDebug],
        previous_quad: np.ndarray | None,
        quad_result: _QuadTrackResult | None,
        metadata: dict[str, Any],
    ) -> None:
        if not self.debug:
            return

        canvas = image.copy()
        self._draw_optical_flow_debug_summary(canvas, marker_debug, quad_result, metadata)

        if previous_quad is not None:
            if quad_result is not None and quad_result.succeeded and quad_result.corners is not None:
                self._draw_quad(canvas, quad_result.corners, (255, 0, 0), label="quad")
            else:
                reason = "missing" if quad_result is None else quad_result.reason
                self._draw_quad(
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
                self._draw_quad(canvas, item.result.corners, (0, 255, 0), label=str(item.marker_id))
                continue
            failed_label = f"{item.marker_id} failed:{item.result.reason}"
            self._draw_quad(
                canvas,
                item.previous_corners,
                (0, 165, 255),
                label=failed_label,
                dashed=True,
            )
            if item.result.corners is not None:
                self._draw_quad(
                    canvas,
                    item.result.corners,
                    (0, 0, 255),
                    label=f"{item.marker_id} rejected",
                )

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self._optical_flow_debug_path(metadata)
        if not cv2.imwrite(str(path), canvas):
            logger.warning("Failed to write optical-flow debug image: %s", path)

    def _write_detector_debug_image(
        self,
        image: np.ndarray,
        seeded_markers: list[_TrackedMarker],
        source_quad: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> None:
        if not self.debug:
            return

        canvas = image.copy()
        self._draw_detector_debug_summary(canvas, seeded_markers, source_quad, metadata)
        if source_quad is not None:
            self._draw_quad(canvas, source_quad, (255, 0, 0), label="quad detector")
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
            self._draw_quad(canvas, marker.corners, (0, 255, 255), label=str(marker.marker_id))

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self._optical_flow_debug_path(metadata)
        if not cv2.imwrite(str(path), canvas):
            logger.warning("Failed to write detector refresh debug image: %s", path)

    @staticmethod
    def _draw_detector_debug_summary(
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
        y = 30
        for line in lines:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            y += 28

    @staticmethod
    def _tracking_failure_summary(
        marker_debug: list[_MarkerTrackDebug],
        quad_result: _QuadTrackResult | None,
    ) -> str:
        if marker_debug:
            reasons = sorted({item.result.reason for item in marker_debug})
            return ",".join(reasons)
        if quad_result is not None:
            return quad_result.reason
        return "no_trackable_markers"

    @staticmethod
    def _draw_optical_flow_debug_summary(
        image: np.ndarray,
        marker_debug: list[_MarkerTrackDebug],
        quad_result: _QuadTrackResult | None,
        metadata: dict[str, Any],
    ) -> None:
        tracked_count = sum(1 for item in marker_debug if item.result.succeeded)
        attempted_count = len(marker_debug)
        tracking_source = metadata.get("tracking_source", "optical_flow")
        confidence = metadata.get("tracking_confidence")
        confidence_text = "" if confidence is None else f" confidence {float(confidence):.3f}"
        quad_text = "quad n/a"
        if quad_result is not None:
            quad_text = "quad ok" if quad_result.succeeded else f"quad failed:{quad_result.reason}"
        lines = [
            f"{tracking_source}: markers {tracked_count}/{attempted_count}{confidence_text}",
            quad_text,
        ]
        if tracking_source == "optical_flow_failed":
            lines.append(f"fallback reason: {metadata.get('tracking_failure_reason', 'unknown')}")

        y = 30
        for line in lines:
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            y += 28

    @staticmethod
    def _draw_quad(
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
                OpticalFlowMarkerTrackingModule._draw_dashed_line(image, p0, p1, color, 2)
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
        origin = tuple(int(value) for value in points[0] + np.array([8, 18], dtype=np.int32))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

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
            detections=detections,
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
        corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
        support_points = self._quad_support_points(corners)
        previous_points = support_points.reshape(-1, 1, 2)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
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
            None,
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
            & (errors.reshape(-1) <= self.max_forward_error)
            & (backtrack_errors <= self.max_backtrack_error)
        )
        min_valid_points = max(4, int(np.ceil(len(support_points) * 0.25)))
        if int(np.count_nonzero(valid)) < min_valid_points:
            return _QuadTrackResult(None, None, "too_few_valid_points")

        previous_valid = support_points[valid]
        next_valid = next_points.reshape(-1, 2)[valid]
        tracked = self._transform_corners_from_flow(corners, previous_valid, next_valid)
        if tracked is None:
            return _QuadTrackResult(None, None, "transform_failed")
        if not self._quad_in_frame(tracked, image_shape):
            return _QuadTrackResult(tracked.astype(np.float32), None, "out_of_frame")
        if polygon_area(tracked) < self.min_marker_area:
            return _QuadTrackResult(tracked.astype(np.float32), None, "below_min_area")

        valid_errors = errors.reshape(-1)[valid]
        valid_backtrack_errors = backtrack_errors[valid]
        valid_ratio = float(np.count_nonzero(valid)) / max(float(len(support_points)), 1.0)
        forward_confidence = 1.0 - min(float(np.mean(valid_errors)), self.max_forward_error) / max(self.max_forward_error, 1e-6)
        backtrack_confidence = 1.0 - min(float(np.mean(valid_backtrack_errors)), self.max_backtrack_error) / max(
            self.max_backtrack_error,
            1e-6,
        )
        confidence = max(0.0, min(1.0, valid_ratio * (forward_confidence + backtrack_confidence) * 0.5))
        return _QuadTrackResult(tracked.astype(np.float32), confidence, "tracked")

    @staticmethod
    def _quad_support_points(corners: np.ndarray, grid_size: int = 5) -> np.ndarray:
        tl, tr, br, bl = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        samples = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
        points = []
        for v in samples:
            for u in samples:
                top = (1.0 - u) * tl + u * tr
                bottom = (1.0 - u) * bl + u * br
                points.append((1.0 - v) * top + v * bottom)
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _transform_corners_from_flow(
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
            if homography is not None and inliers is not None and int(np.count_nonzero(inliers)) >= 4:
                transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography)
                return transformed.reshape(4, 2).astype(np.float32)

        if len(previous_points) >= 3:
            affine, inliers = cv2.estimateAffinePartial2D(
                previous_points.astype(np.float32),
                next_points.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if affine is not None and (inliers is None or int(np.count_nonzero(inliers)) >= 3):
                transformed = cv2.transform(corners.reshape(-1, 1, 2), affine)
                return transformed.reshape(4, 2).astype(np.float32)

        if len(previous_points) >= 1:
            delta = np.median(next_points - previous_points, axis=0).astype(np.float32)
            return (corners + delta).astype(np.float32)
        return None

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
    def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        source_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(source_points, homography)
        return transformed.reshape(-1, 2).astype(np.float32)

    @staticmethod
    def _homography(metadata: dict[str, Any]) -> np.ndarray | None:
        raw_homography = metadata.get("cutout_to_source_homography")
        if raw_homography is None:
            logger.warning("Cannot seed optical flow tracking without cutout-to-source homography")
            return None
        homography = np.asarray(raw_homography, dtype=np.float32)
        if homography.shape != (3, 3):
            logger.warning("Cannot seed optical flow tracking with invalid homography shape: %s", homography.shape)
            return None
        return homography

    @staticmethod
    def _source_quad(metadata: dict[str, Any]) -> np.ndarray | None:
        raw_quad = metadata.get("source_quad")
        if raw_quad is None:
            raw_quad = metadata.get("quad")
        if raw_quad is None:
            return None
        quad = np.asarray(raw_quad, dtype=np.float32)
        if quad.shape != (4, 2):
            return None
        return quad

    @staticmethod
    def _quad_in_frame(points: np.ndarray, image_shape: tuple[int, ...]) -> bool:
        height, width = image_shape[:2]
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return bool(
            np.all(np.isfinite(points))
            and np.all(points[:, 0] >= 0)
            and np.all(points[:, 0] <= width - 1)
            and np.all(points[:, 1] >= 0)
            and np.all(points[:, 1] <= height - 1)
        )
