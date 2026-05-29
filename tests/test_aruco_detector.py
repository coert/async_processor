from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from src import (
    ArucoDetectionModule,
    ArucoDetectionResult,
    AsyncProcessor,
    Message,
    VideoFrame,
)


def make_aruco_test_image(marker_id: int = 23) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    image[40:200, 40:200] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def test_aruco_detection_module_detects_expected_marker_id() -> None:
    async def scenario() -> None:
        image = make_aruco_test_image(marker_id=23)
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            dictionary_name="DICT_6X6_1000",
        )

        routed = await module.process(Message(image), AsyncProcessor())

        assert routed is not None
        assert routed.destination == "detections"
        result = routed.message.payload
        assert isinstance(result, ArucoDetectionResult)
        assert result.image is image
        assert [detection.marker_id for detection in result.detections] == [23]
        assert result.detections[0].corners.shape == (4, 2)
        assert routed.message.metadata["dictionary_name"] == "DICT_6X6_1000"
        assert routed.message.metadata["marker_count"] == 1
        assert routed.message.metadata["marker_ids"] == [23]
        assert routed.message.metadata["input_border_pixels"] == 16
        assert routed.message.metadata["detection_passes"] == {23: "raw"}

    asyncio.run(scenario())


def test_aruco_detection_module_merges_raw_and_padded_detections(caplog) -> None:
    async def scenario() -> None:
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            input_border_pixels=16,
        )

        def fake_detect(
            candidate: np.ndarray,
        ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
            if candidate.shape == image.shape:
                return (
                    [np.array([[[1, 1], [5, 1], [5, 5], [1, 5]]], dtype=np.float32)],
                    np.array([[1]], dtype=np.int32),
                    [],
                )
            return (
                [
                    np.array(
                        [[[17, 17], [21, 17], [21, 21], [17, 21]]], dtype=np.float32
                    ),
                    np.array(
                        [[[26, 26], [30, 26], [30, 30], [26, 30]]], dtype=np.float32
                    ),
                ],
                np.array([[1], [2]], dtype=np.int32),
                [],
            )

        module._detect = fake_detect  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger="src.modules.aruco_detector"):
            routed = await module.process(Message(image), AsyncProcessor())

        assert routed is not None
        assert "ArUco marker id mismatch" not in caplog.text
        assert routed.message.metadata["marker_ids"] == [1, 2]
        assert routed.message.metadata["detection_passes"] == {1: "raw", 2: "padded"}
        assert np.array_equal(
            routed.message.payload.detections[0].corners,
            np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=np.float32),
        )
        assert np.array_equal(
            routed.message.payload.detections[1].corners,
            np.array([[10, 10], [14, 10], [14, 14], [10, 14]], dtype=np.float32),
        )

    asyncio.run(scenario())


def test_aruco_detection_module_warns_when_same_quad_has_different_ids(caplog) -> None:
    async def scenario() -> None:
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            input_border_pixels=16,
        )

        def fake_detect(
            candidate: np.ndarray,
        ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
            if candidate.shape == image.shape:
                return (
                    [np.array([[[1, 1], [5, 1], [5, 5], [1, 5]]], dtype=np.float32)],
                    np.array([[1]], dtype=np.int32),
                    [],
                )
            return (
                [
                    np.array(
                        [[[17, 17], [21, 17], [21, 21], [17, 21]]], dtype=np.float32
                    )
                ],
                np.array([[2]], dtype=np.int32),
                [],
            )

        module._detect = fake_detect  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger="src.modules.aruco_detector"):
            routed = await module.process(Message(image), AsyncProcessor())

        assert routed is not None
        assert "ArUco marker id mismatch for same detection quad" in caplog.text

    asyncio.run(scenario())


def test_aruco_detection_module_can_disable_input_border() -> None:
    async def scenario() -> None:
        image = make_aruco_test_image(marker_id=5)
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            input_border_pixels=0,
        )

        routed = await module.process(Message(image), AsyncProcessor())

        assert routed is not None
        assert routed.message.payload.image is image
        assert routed.message.metadata["input_border_pixels"] == 0

    asyncio.run(scenario())


def test_aruco_detection_module_rejects_negative_input_border() -> None:
    try:
        ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            input_border_pixels=-1,
        )
    except ValueError as exc:
        assert "input_border_pixels" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_aruco_detection_module_preserves_video_frame_metadata() -> None:
    async def scenario() -> None:
        frame = VideoFrame(
            image=make_aruco_test_image(marker_id=7),
            frame_index=42,
            timestamp_seconds=1.4,
            loop_count=2,
        )
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
        )

        routed = await module.process(
            Message(frame, metadata={"source": "test"}), AsyncProcessor()
        )

        assert routed is not None
        assert routed.message.metadata["source"] == "test"
        assert routed.message.metadata["frame_index"] == 42
        assert routed.message.metadata["timestamp_seconds"] == 1.4
        assert routed.message.metadata["loop_count"] == 2
        assert routed.message.metadata["marker_ids"] == [7]

    asyncio.run(scenario())


def test_aruco_detection_module_drops_frame_without_markers() -> None:
    async def scenario() -> None:
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
        )
        blank = np.full((120, 160, 3), 127, dtype=np.uint8)

        routed = await module.process(Message(blank), AsyncProcessor())

        assert routed is None

    asyncio.run(scenario())


def test_aruco_detection_module_writes_debug_images(tmp_path: Path) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = ArucoDetectionModule(
            name="aruco",
            input_queue="frames",
            output_queue="detections",
            debug=True,
            debug_dir=debug_dir,
        )
        frame = VideoFrame(
            image=make_aruco_test_image(),
            frame_index=12,
            timestamp_seconds=0.48,
            loop_count=0,
        )

        routed = await module.process(Message(frame), AsyncProcessor())

        assert routed is not None
        for name in (
            "aruco_input.png",
            "aruco_detected_markers_0012.png",
            "aruco_rejected_candidates_0012.png",
        ):
            path = debug_dir / name
            assert path.exists()
            assert cv2.imread(str(path)) is not None
        assert cv2.imread(str(debug_dir / "aruco_input.png")).shape == (240, 240, 3)
        assert cv2.imread(str(debug_dir / "aruco_detected_markers_0012.png")).shape == (
            272,
            272,
            3,
        )
        assert cv2.imread(
            str(debug_dir / "aruco_rejected_candidates_0012.png")
        ).shape == (272, 272, 3)
        assert not (debug_dir / "aruco_detected_markers.png").exists()
        assert not (debug_dir / "aruco_rejected_candidates.png").exists()
        assert not (debug_dir / "aruco_padded_input.png").exists()

    asyncio.run(scenario())


def test_aruco_detection_module_routes_detection_result_to_configured_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("frames")
        processor.create_queue("detections")
        processor.register_module(
            ArucoDetectionModule(
                name="aruco",
                input_queue="frames",
                output_queue="detections",
            )
        )

        await processor.start()
        await processor.submit("frames", make_aruco_test_image(marker_id=11))

        result = await asyncio.wait_for(processor.queue("detections").get(), timeout=1)
        assert isinstance(result.payload, ArucoDetectionResult)
        assert [detection.marker_id for detection in result.payload.detections] == [11]
        assert result.metadata["marker_ids"] == [11]
        processor.queue("detections").task_done()
        await processor.stop()

    asyncio.run(scenario())
