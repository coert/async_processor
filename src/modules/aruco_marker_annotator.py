from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..messages import Message, RoutedMessage
from .aruco_detector import ArucoDetectionResult, ArucoMarkerDetection
from .base import BaseModule, ModuleContext
from .image_enhancer import validate_color_image

logger = logging.getLogger(__name__)

ORIGINAL_FRAME_METADATA_KEY = "original_frame_image"
SOURCE_FRAME_METADATA_KEY = "source_frame_image"
CUTOUT_TO_SOURCE_HOMOGRAPHY_METADATA_KEY = "cutout_to_source_homography"
MARKER_TEMPLATE_PREFIX = "6x6_1000"


class ArucoMarkerAnnotationModule(BaseModule[ArucoDetectionResult]):
    def __init__(
        self,
        name: str,
        input_queue: str,
        output_queue: str,
        *,
        debug: bool = False,
        debug_dir: Path | str = Path("data/debug"),
        marker_template_dir: Path | str = Path("data/aruco/6x6_1000"),
        template_marker_size: int = 64,
        template_margin_pixels: int = 12,
    ) -> None:
        if not output_queue:
            raise ValueError("Module output_queue cannot be empty.")
        if template_marker_size <= 0:
            raise ValueError("template_marker_size must be greater than zero.")
        if template_margin_pixels < 0:
            raise ValueError("template_margin_pixels cannot be negative.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_queue = output_queue
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.marker_template_dir = Path(marker_template_dir)
        self.template_marker_size = template_marker_size
        self.template_margin_pixels = template_margin_pixels
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _marker_center(corners: np.ndarray) -> np.ndarray:
        return np.mean(np.asarray(corners, dtype=np.float32).reshape(4, 2), axis=0)

    @staticmethod
    def _marker_height(corners: np.ndarray) -> float:
        marker_corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        left = float(np.linalg.norm(marker_corners[3] - marker_corners[0]))
        right = float(np.linalg.norm(marker_corners[2] - marker_corners[1]))
        return max((left + right) / 2.0, 1.0)

    @staticmethod
    def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
        source_points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(source_points, homography)
        return transformed.reshape(-1, 2)

    @staticmethod
    def _text_origin(
        text: str,
        center: tuple[int, int],
        font_face: int,
        font_scale: float,
        thickness: int,
    ) -> tuple[int, int]:
        text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        x = int(round(center[0] - text_size[0] / 2))
        y = int(round(center[1] + (text_size[1] - baseline) / 2))
        return x, y

    def _draw_marker_id(
        self,
        image: np.ndarray,
        marker_id: int,
        center: np.ndarray,
    ) -> None:
        text = str(marker_id)
        center_xy = tuple(int(round(value)) for value in center)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        fill_thickness = 2
        origin = self._text_origin(
            text, center_xy, font_face, font_scale, fill_thickness
        )
        cv2.putText(
            image,
            text,
            origin,
            font_face,
            font_scale,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            font_face,
            font_scale,
            (255, 255, 255),
            fill_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _frame_info_text(metadata: dict[str, Any]) -> str | None:
        frame_index = metadata.get("frame_index")
        timestamp_seconds = metadata.get("timestamp_seconds")
        if frame_index is None and timestamp_seconds is None:
            return None
        frame_text = "Frame ?" if frame_index is None else f"Frame {int(frame_index)}"
        time_text = (
            "?.???s"
            if timestamp_seconds is None
            else f"{float(timestamp_seconds):.3f}s"
        )
        return f"{frame_text} | {time_text}"

    def _draw_frame_info_box(self, image: np.ndarray, metadata: dict[str, Any]) -> None:
        text = self._frame_info_text(metadata)
        if text is None:
            return

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        padding_x = 10
        padding_y = 8
        margin = 12
        vertical_offset = 100
        text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        box_width = text_size[0] + (2 * padding_x)
        box_height = text_size[1] + baseline + (2 * padding_y)
        x0 = margin
        y0 = max(0, image.shape[0] - margin - box_height - vertical_offset)
        x1 = min(image.shape[1] - 1, x0 + box_width)
        y1 = min(image.shape[0] - 1, y0 + box_height)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 1)
        origin = (x0 + padding_x, y0 + padding_y + text_size[1])
        cv2.putText(
            image,
            text,
            origin,
            font_face,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _write_debug_image(self, image: np.ndarray, metadata: dict[str, Any]) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_image = image.copy()
        path = self.debug_dir / "aruco_annotated_frame.png"
        if not cv2.imwrite(str(path), debug_image):
            logger.warning("Failed to write ArUco annotation debug image: %s", path)

    def _source_image(self, metadata: dict[str, Any]) -> np.ndarray | None:
        image = metadata.get(ORIGINAL_FRAME_METADATA_KEY)
        if image is None:
            image = metadata.get(SOURCE_FRAME_METADATA_KEY)
        if image is None:
            logger.warning(
                "Dropping ArUco annotation without source frame image metadata"
            )
            return None
        if not isinstance(image, np.ndarray):
            logger.warning("Dropping ArUco annotation with non-image source metadata")
            return None
        return image

    def _homography(self, metadata: dict[str, Any]) -> np.ndarray | None:
        raw_homography = metadata.get(CUTOUT_TO_SOURCE_HOMOGRAPHY_METADATA_KEY)
        if raw_homography is None:
            logger.warning(
                "Dropping ArUco annotation without cutout-to-source homography metadata"
            )
            return None
        homography = np.asarray(raw_homography, dtype=np.float32)
        if homography.shape != (3, 3):
            logger.warning(
                "Dropping ArUco annotation with invalid homography shape: %s",
                homography.shape,
            )
            return None
        return homography

    def _infer_marker_grid(
        self,
        detections: list[ArucoMarkerDetection],
    ) -> list[list[int | None]]:
        if not detections:
            return []

        marker_items = [
            {
                "marker_id": detection.marker_id,
                "center": self._marker_center(detection.corners),
                "height": self._marker_height(detection.corners),
            }
            for detection in detections
        ]
        median_height = float(np.median([item["height"] for item in marker_items]))
        row_tolerance = max(1.0, median_height * 0.5)

        rows: list[list[dict[str, Any]]] = []
        for item in sorted(marker_items, key=lambda value: float(value["center"][1])):
            center_y = float(item["center"][1])
            for row in rows:
                row_y = float(
                    np.mean([float(existing["center"][1]) for existing in row])
                )
                if abs(center_y - row_y) <= row_tolerance:
                    row.append(item)
                    break
            else:
                rows.append([item])

        rows.sort(
            key=lambda row: float(np.mean([float(item["center"][1]) for item in row]))
        )
        sorted_rows = [
            sorted(row, key=lambda item: float(item["center"][0])) for row in rows
        ]
        max_columns = max(len(row) for row in sorted_rows)
        return [
            [int(item["marker_id"]) for item in row] + [None] * (max_columns - len(row))
            for row in sorted_rows
        ]

    def _template_path(self, marker_id: int) -> Path:
        return (
            self.marker_template_dir / f"{MARKER_TEMPLATE_PREFIX}_{marker_id:04d}.png"
        )

    def _blank_template(self, cell_size: int) -> np.ndarray:
        return np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)

    def _load_marker_template(self, marker_id: int, cell_size: int) -> np.ndarray:
        path = self._template_path(marker_id)
        marker = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if marker is None:
            logger.warning("Missing or unreadable ArUco marker template: %s", path)
            return self._blank_template(cell_size)
        if marker.shape[:2] != (cell_size, cell_size):
            marker = cv2.resize(
                marker, (cell_size, cell_size), interpolation=cv2.INTER_NEAREST
            )
        return marker

    def _cell_size_for_frame(
        self,
        frame_shape: tuple[int, ...],
        rows: int,
        columns: int,
    ) -> int:
        frame_height, frame_width = frame_shape[:2]
        available_width = frame_width - (2 * self.template_margin_pixels)
        available_height = frame_height - (2 * self.template_margin_pixels)
        if available_width <= 0 or available_height <= 0:
            return 0
        return max(
            1,
            min(
                self.template_marker_size,
                available_width // columns,
                available_height // rows,
            ),
        )

    def _compose_template_grid(
        self,
        marker_grid: list[list[int | None]],
        cell_size: int,
    ) -> np.ndarray | None:
        if not marker_grid or cell_size <= 0:
            return None

        rows = []
        for marker_row in marker_grid:
            cells = [
                self._blank_template(cell_size)
                if marker_id is None
                else self._load_marker_template(marker_id, cell_size)
                for marker_id in marker_row
            ]
            rows.append(np.hstack(cells))
        return np.vstack(rows)

    def _draw_template_grid(
        self,
        image: np.ndarray,
        detections: list[ArucoMarkerDetection],
    ) -> tuple[int, int] | None:
        marker_grid = self._infer_marker_grid(detections)
        if not marker_grid:
            return None

        rows = len(marker_grid)
        columns = max(len(row) for row in marker_grid)
        cell_size = self._cell_size_for_frame(image.shape, rows, columns)
        grid_image = self._compose_template_grid(marker_grid, cell_size)
        if grid_image is None:
            return None

        grid_height, grid_width = grid_image.shape[:2]
        x0 = image.shape[1] - self.template_margin_pixels - grid_width
        y0 = image.shape[0] - self.template_margin_pixels - grid_height
        if x0 < 0 or y0 < 0:
            return None
        image[y0 : y0 + grid_height, x0 : x0 + grid_width] = grid_image
        return rows, columns

    async def process(
        self,
        message: Message[ArucoDetectionResult],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray] | None:
        result = message.payload
        metadata: dict[str, Any] = dict(message.metadata)
        source_image = self._source_image(metadata)
        homography = self._homography(metadata)
        if source_image is None or homography is None:
            return None

        validate_color_image(source_image)
        annotated = source_image.copy()
        template_grid_shape = None
        if result.detections:
            centers = np.asarray(
                [
                    self._marker_center(detection.corners)
                    for detection in result.detections
                ],
                dtype=np.float32,
            )
            frame_centers = self._transform_points(centers, homography)
            for detection, center in zip(result.detections, frame_centers):
                self._draw_marker_id(annotated, detection.marker_id, center)
            template_grid_shape = self._draw_template_grid(annotated, result.detections)

        self._draw_frame_info_box(annotated, metadata)

        metadata.update(
            {
                "annotated_marker_count": len(result.detections),
                "annotated_marker_ids": [
                    detection.marker_id for detection in result.detections
                ],
                "template_grid_shape": template_grid_shape,
            }
        )
        self._write_debug_image(annotated, metadata)
        return RoutedMessage(
            destination=self.output_queue,
            message=Message(payload=annotated, metadata=metadata),
        )
