from __future__ import annotations

import asyncio
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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
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
            dictionary_name="DICT_5X5_1000",
        )

        routed = await module.process(Message(image), AsyncProcessor())

        assert routed is not None
        assert routed.destination == "detections"
        result = routed.message.payload
        assert isinstance(result, ArucoDetectionResult)
        assert result.image is image
        assert [detection.marker_id for detection in result.detections] == [23]
        assert result.detections[0].corners.shape == (4, 2)
        assert routed.message.metadata["dictionary_name"] == "DICT_5X5_1000"
        assert routed.message.metadata["marker_count"] == 1
        assert routed.message.metadata["marker_ids"] == [23]

    asyncio.run(scenario())


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

        routed = await module.process(Message(frame, metadata={"source": "test"}), AsyncProcessor())

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

        routed = await module.process(Message(make_aruco_test_image()), AsyncProcessor())

        assert routed is not None
        for name in (
            "aruco_input.png",
            "aruco_detected_markers.png",
            "aruco_rejected_candidates.png",
        ):
            path = debug_dir / name
            assert path.exists()
            assert cv2.imread(str(path)) is not None

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
