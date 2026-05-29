from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path

import joblib
import numpy as np
import cv2
import pytest

import main as app_main
from src.modules.marker_rectifier import core as marker_rectifier_core
from src.modules.marker_rectifier import module as marker_rectifier_module
from src.modules.marker_rectifier.models import EdgeArtifacts, MarkerEvidence

from src.modules.marker_rectifier import (
    Candidate,
    apply_nms,
    quad_iou,
    compute_bw_dominance,
    refine_candidate,
)
from src.modules.image_enhancer import ORIGINAL_FRAME_METADATA_KEY, apply_enhancement

from src import (
    ArucoDetectionResult,
    ArucoMarkerAnnotationModule,
    ArucoMarkerDetection,
    AsyncProcessor,
    BaseModule,
    ColorFormatter,
    DuplicateModuleError,
    DuplicateQueueError,
    FfmpegVideoWriterModule,
    FiniteImageSource,
    FiniteVideoSource,
    FrameRateLoggerModule,
    ImageEnhancementModule,
    ImageFrame,
    ImageSourceError,
    GMMColorMaskModule,
    MarkerRectificationModule,
    OpticalFlowMarkerTrackingModule,
    LoopingImageSource,
    LoopingVideoSource,
    Message,
    ModuleContext,
    ProcessorLoop,
    QueueFanoutModule,
    RoutedMessage,
    SignalStopper,
    UnknownQueueError,
    VideoFrame,
)


TEST_VIDEO_PATH = Path(__file__).parents[1] / "data" / "1-input.mp4"


class DistanceGMM:
    def __init__(self, center: np.ndarray) -> None:
        self.center = center.astype(np.float64)

    def score_samples(self, pixels: np.ndarray) -> np.ndarray:
        delta = pixels.astype(np.float64) - self.center
        return -np.sum(delta * delta, axis=1) / 100.0


def write_synthetic_gmm_model(path: Path, query_bgr: tuple[int, int, int]) -> None:
    query_pixel = np.array([[query_bgr]], dtype=np.uint8)
    query_lab = cv2.cvtColor(query_pixel, cv2.COLOR_BGR2LAB).reshape(3)
    non_query_lab = cv2.cvtColor(
        np.array([[[255, 0, 0]]], dtype=np.uint8),
        cv2.COLOR_BGR2LAB,
    ).reshape(3)
    joblib.dump(
        {
            "query_gmm": DistanceGMM(query_lab),
            "non_query_gmm": DistanceGMM(non_query_lab),
            "query_prior": 0.5,
            "non_query_prior": 0.5,
        },
        path,
    )


class UppercaseModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload("out", message.payload.upper())


class FanoutModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> list[RoutedMessage[str]]:
        return [
            RoutedMessage.from_payload("out_a", f"{message.payload}:a"),
            RoutedMessage.from_payload("out_b", f"{message.payload}:b"),
        ]


class MissingRouteModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload("missing", message.payload)


class SinkModule(BaseModule[str]):
    def __init__(self, name: str, input_queue: str) -> None:
        super().__init__(name, input_queue)
        self.seen: list[str] = []

    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> None:
        self.seen.append(message.payload)
        return None


class ListSource:
    def __init__(self, items: list[str]) -> None:
        self.items = items

    async def poll(self) -> str | None:
        if not self.items:
            return None
        return self.items.pop(0)


def test_queue_creation_and_duplicates() -> None:
    processor = AsyncProcessor()

    processor.create_queue("in")

    with pytest.raises(DuplicateQueueError):
        processor.create_queue("in")


def test_duplicate_module_and_input_queue_validation() -> None:
    processor = AsyncProcessor()
    processor.create_queue("in")
    processor.create_queue("other")
    processor.register_module(SinkModule("sink", "in"))

    with pytest.raises(DuplicateModuleError):
        processor.register_module(SinkModule("sink", "other"))

    with pytest.raises(DuplicateModuleError):
        processor.register_module(SinkModule("second", "in"))

    with pytest.raises(UnknownQueueError):
        processor.register_module(SinkModule("missing", "missing"))


def test_module_consumes_dedicated_queue_and_routes_output() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out")
        processor.register_module(UppercaseModule("upper", "in"))

        await processor.start()
        await processor.submit("in", "hello")

        result = await asyncio.wait_for(processor.queue("out").get(), timeout=1)
        assert result.payload == "HELLO"
        processor.queue("out").task_done()

        await processor.stop()

    asyncio.run(scenario())


def test_multiple_outputs_from_one_input() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out_a")
        processor.create_queue("out_b")
        processor.register_module(FanoutModule("fanout", "in"))

        await processor.start()
        await processor.submit("in", "event")

        result_a = await asyncio.wait_for(processor.queue("out_a").get(), timeout=1)
        result_b = await asyncio.wait_for(processor.queue("out_b").get(), timeout=1)

        assert result_a.payload == "event:a"
        assert result_b.payload == "event:b"

        processor.queue("out_a").task_done()
        processor.queue("out_b").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_graceful_shutdown_without_hanging() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        module = SinkModule("sink", "in")
        processor.register_module(module)

        await processor.start()
        await processor.submit("in", "one")
        await asyncio.wait_for(processor.queue("in").join(), timeout=1)
        await asyncio.wait_for(processor.stop(), timeout=1)

        assert module.seen == ["one"]

    asyncio.run(scenario())


def test_unknown_route_target_is_surfaced_by_wait() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.register_module(MissingRouteModule("bad-route", "in"))

        await processor.start()
        await processor.submit("in", "hello")

        with pytest.raises(UnknownQueueError):
            await asyncio.wait_for(processor.wait(), timeout=1)

    asyncio.run(scenario())


def test_loop_polls_source_and_submits_to_input_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.create_queue("out")
        processor.register_module(UppercaseModule("upper", "in"))
        source = ListSource(["frame"])
        stop_event = asyncio.Event()
        runner = ProcessorLoop(
            processor,
            input_queue="in",
            source=source,
            poll_interval=0.001,
        )

        task = asyncio.create_task(runner.run(stop_event=stop_event))
        result = await asyncio.wait_for(processor.queue("out").get(), timeout=1)
        assert result.payload == "FRAME"
        processor.queue("out").task_done()

        stop_event.set()
        await asyncio.wait_for(task, timeout=1)
        assert processor.is_running is False

    asyncio.run(scenario())


def test_empty_loop_stops_cleanly_when_stop_event_is_set() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        stop_event = asyncio.Event()
        runner = ProcessorLoop(processor, poll_interval=0.001)

        task = asyncio.create_task(runner.run(stop_event=stop_event))
        await asyncio.sleep(0)
        assert processor.is_running is True

        stop_event.set()
        await asyncio.wait_for(task, timeout=1)
        assert processor.is_running is False

    asyncio.run(scenario())


def test_loop_surfaces_module_failures_and_stops_cleanly() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("in")
        processor.register_module(MissingRouteModule("bad-route", "in"))
        source = ListSource(["hello"])
        runner = ProcessorLoop(
            processor,
            input_queue="in",
            source=source,
            poll_interval=0.001,
        )

        with pytest.raises(UnknownQueueError):
            await asyncio.wait_for(runner.run(stop_event=asyncio.Event()), timeout=1)

        assert processor.is_running is False

    asyncio.run(scenario())


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is unavailable")
def test_signal_stopper_sets_stop_event_from_signal() -> None:
    async def scenario() -> None:
        async with SignalStopper(signals=(signal.SIGUSR1,)) as stop_event:
            os.kill(os.getpid(), signal.SIGUSR1)
            await asyncio.wait_for(stop_event.wait(), timeout=1)

    asyncio.run(scenario())


def test_color_formatter_colors_expected_levels() -> None:
    formatter = ColorFormatter("%(levelname)s:%(message)s", use_colors=True)

    for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname=__file__,
            lineno=1,
            msg="message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "\033[" in formatted
        assert logging.getLevelName(level) in formatted
        assert formatted.endswith("\033[0m:message")


def test_color_formatter_can_disable_colors() -> None:
    formatter = ColorFormatter("%(levelname)s:%(message)s", use_colors=False)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="message",
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == "INFO:message"


def test_looping_video_source_reads_and_loops_test_video() -> None:
    async def scenario() -> None:
        source = LoopingVideoSource(TEST_VIDEO_PATH, realtime=False)
        try:
            first_frame = await source.poll()
            assert first_frame.frame_index == 0
            assert first_frame.loop_count == 0
            assert first_frame.image.shape[:2] == (source.height, source.width)

            looped_frame = first_frame
            for _ in range(source.frame_count):
                looped_frame = await source.poll()

            assert looped_frame.frame_index == 0
            assert looped_frame.loop_count == 1
        finally:
            source.close()

        assert source.is_open is False

    asyncio.run(scenario())


def test_finite_video_source_reads_one_pass_without_looping() -> None:
    async def scenario() -> None:
        source = FiniteVideoSource(TEST_VIDEO_PATH, realtime=False)
        try:
            frames = []
            while True:
                frame = await source.poll()
                if frame is None:
                    break
                frames.append(frame)

            assert len(frames) == source.frame_count
            assert frames[0].frame_index == 0
            assert frames[0].loop_count == 0
            assert frames[-1].loop_count == 0
            assert await source.poll() is None
        finally:
            source.close()

    asyncio.run(scenario())


def write_test_source_image(path: Path, color: tuple[int, int, int]) -> np.ndarray:
    image = np.full((4, 5, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def test_looping_image_source_reads_single_image_and_loops(tmp_path: Path) -> None:
    async def scenario() -> None:
        image_path = tmp_path / "one.png"
        expected = write_test_source_image(image_path, (1, 2, 3))
        source = LoopingImageSource(image_path)

        first_frame = await source.poll()
        second_frame = await source.poll()

        assert isinstance(first_frame, ImageFrame)
        assert first_frame.frame_index == 0
        assert first_frame.loop_count == 0
        assert first_frame.path == image_path
        assert np.array_equal(first_frame.image, expected)
        assert second_frame.frame_index == 0
        assert second_frame.loop_count == 1

    asyncio.run(scenario())


def test_looping_image_source_reads_multiple_images_in_input_order(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first_path = tmp_path / "b.png"
        second_path = tmp_path / "a.png"
        first = write_test_source_image(first_path, (10, 20, 30))
        second = write_test_source_image(second_path, (40, 50, 60))
        source = LoopingImageSource([first_path, second_path])

        first_frame = await source.poll()
        second_frame = await source.poll()
        looped_frame = await source.poll()

        assert first_frame.path == first_path
        assert np.array_equal(first_frame.image, first)
        assert second_frame.path == second_path
        assert np.array_equal(second_frame.image, second)
        assert looped_frame.path == first_path
        assert looped_frame.loop_count == 1

    asyncio.run(scenario())


def test_finite_image_source_reads_one_pass_without_looping(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_path = tmp_path / "b.png"
        second_path = tmp_path / "a.png"
        first = write_test_source_image(first_path, (10, 20, 30))
        second = write_test_source_image(second_path, (40, 50, 60))
        source = FiniteImageSource([first_path, second_path])

        first_frame = await source.poll()
        second_frame = await source.poll()
        exhausted = await source.poll()

        assert first_frame is not None
        assert second_frame is not None
        assert first_frame.path == first_path
        assert np.array_equal(first_frame.image, first)
        assert first_frame.loop_count == 0
        assert second_frame.path == second_path
        assert np.array_equal(second_frame.image, second)
        assert second_frame.loop_count == 0
        assert exhausted is None

    asyncio.run(scenario())


def test_looping_image_source_expands_globs_in_sorted_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        b_path = tmp_path / "b.png"
        a_path = tmp_path / "a.png"
        write_test_source_image(b_path, (10, 20, 30))
        write_test_source_image(a_path, (40, 50, 60))
        source = LoopingImageSource(str(tmp_path / "*.png"))

        first_frame = await source.poll()
        second_frame = await source.poll()

        assert first_frame.path == a_path
        assert second_frame.path == b_path

    asyncio.run(scenario())


def test_looping_image_source_expands_padded_numeric_range(tmp_path: Path) -> None:
    expected_paths = [tmp_path / f"frame_{index:04d}.jpg" for index in range(6, 11)]
    for index, image_path in enumerate(expected_paths):
        write_test_source_image(image_path, (index, index + 1, index + 2))

    source = LoopingImageSource(str(tmp_path / "frame_[0006-0010].jpg"))

    assert list(source.paths) == expected_paths


def test_looping_image_source_uses_filename_number_as_frame_index(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first_path = tmp_path / "frame_0100.jpg"
        second_path = tmp_path / "frame_0101.jpg"
        write_test_source_image(first_path, (1, 2, 3))
        write_test_source_image(second_path, (4, 5, 6))

        source = LoopingImageSource(str(tmp_path / "frame_[0100-0101].jpg"))

        first_frame = await source.poll()
        second_frame = await source.poll()

        assert first_frame.frame_index == 100
        assert second_frame.frame_index == 101

    asyncio.run(scenario())


def test_looping_image_source_expands_unpadded_numeric_range(tmp_path: Path) -> None:
    expected_paths = [tmp_path / f"frame_{index}.jpg" for index in range(6, 11)]
    for index, image_path in enumerate(expected_paths):
        write_test_source_image(image_path, (index, index + 1, index + 2))

    source = LoopingImageSource(str(tmp_path / "frame_[6-10].jpg"))

    assert list(source.paths) == expected_paths


def test_looping_image_source_uses_widest_numeric_range_endpoint_for_padding(
    tmp_path: Path,
) -> None:
    expected_paths = [tmp_path / f"frame_{index:03d}.jpg" for index in range(6, 11)]
    for index, image_path in enumerate(expected_paths):
        write_test_source_image(image_path, (index, index + 1, index + 2))

    source = LoopingImageSource(str(tmp_path / "frame_[006-10].jpg"))

    assert list(source.paths) == expected_paths


def test_looping_image_source_skips_missing_files_in_numeric_range(
    tmp_path: Path,
) -> None:
    expected_paths = [
        tmp_path / "frame_0006.jpg",
        tmp_path / "frame_0008.jpg",
        tmp_path / "frame_0010.jpg",
    ]
    for index, image_path in enumerate(expected_paths):
        write_test_source_image(image_path, (index, index + 1, index + 2))

    source = LoopingImageSource(str(tmp_path / "frame_[0006-0010].jpg"))

    assert list(source.paths) == expected_paths


def test_looping_image_source_skips_unreadable_files_in_numeric_range(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "frame_0006.jpg"
    unreadable_path = tmp_path / "frame_0007.jpg"
    last_path = tmp_path / "frame_0008.jpg"
    write_test_source_image(first_path, (1, 2, 3))
    unreadable_path.write_text("not an image")
    write_test_source_image(last_path, (4, 5, 6))

    source = LoopingImageSource(str(tmp_path / "frame_[0006-0008].jpg"))

    assert list(source.paths) == [first_path, last_path]


def test_looping_image_source_raises_when_numeric_range_has_no_readable_images(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ImageSourceError, match="No readable image inputs matched numeric range"
    ):
        LoopingImageSource(str(tmp_path / "frame_[0006-0010].jpg"))


def test_looping_image_source_raises_for_descending_numeric_range(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImageSourceError, match="Numeric range cannot descend"):
        LoopingImageSource(str(tmp_path / "frame_[0010-0006].jpg"))


def test_looping_image_source_raises_for_malformed_numeric_range(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImageSourceError, match="exactly one numeric range"):
        LoopingImageSource(str(tmp_path / "frame_[0006-0010.jpg"))


def test_looping_image_source_raises_for_numeric_range_with_unsupported_extension(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImageSourceError, match="Unsupported image input extension"):
        LoopingImageSource(str(tmp_path / "frame_[0006-0010].txt"))


def test_looping_image_source_raises_for_unmatched_glob(tmp_path: Path) -> None:
    with pytest.raises(ImageSourceError, match="No image inputs matched"):
        LoopingImageSource(str(tmp_path / "*.png"))


def test_looping_image_source_raises_for_empty_input_set() -> None:
    with pytest.raises(ImageSourceError, match="No image inputs matched"):
        LoopingImageSource([])


def test_looping_image_source_raises_for_unreadable_image(tmp_path: Path) -> None:
    async def scenario() -> None:
        image_path = tmp_path / "bad.jpg"
        image_path.write_text("not an image")
        source = LoopingImageSource(image_path)

        with pytest.raises(ImageSourceError, match="Could not read image file"):
            await source.poll()

    asyncio.run(scenario())


def test_create_input_source_uses_video_source_for_single_video_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class SpyVideoSource:
        def __init__(self, path: Path, *, realtime: bool) -> None:
            captured["path"] = path
            captured["realtime"] = realtime

    monkeypatch.setattr(app_main, "LoopingVideoSource", SpyVideoSource)

    source = app_main.create_input_source([Path("input.mp4")], realtime=False)

    assert isinstance(source, SpyVideoSource)
    assert captured == {"path": Path("input.mp4"), "realtime": False}


def test_create_input_source_uses_image_source_for_single_image_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class SpyImageSource:
        def __init__(self, paths: list[Path]) -> None:
            captured["paths"] = paths

    monkeypatch.setattr(app_main, "LoopingImageSource", SpyImageSource)

    source = app_main.create_input_source([Path("input.png")], realtime=True)

    assert isinstance(source, SpyImageSource)
    assert captured == {"paths": [Path("input.png")]}


def test_create_input_source_treats_multiple_inputs_as_image_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class SpyImageSource:
        def __init__(self, paths: list[Path]) -> None:
            captured["paths"] = paths

    monkeypatch.setattr(app_main, "LoopingImageSource", SpyImageSource)

    source = app_main.create_input_source(
        [Path("input.mp4"), Path("input.png")], realtime=True
    )

    assert isinstance(source, SpyImageSource)
    assert captured == {"paths": [Path("input.mp4"), Path("input.png")]}


def test_frame_rate_logger_module_logs_processing_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        module = FrameRateLoggerModule(
            name="fps",
            input_queue="frames",
            log_interval_seconds=0,
        )
        frame = VideoFrame(
            image=object(),
            frame_index=0,
            timestamp_seconds=0.0,
            loop_count=0,
        )

        with caplog.at_level(logging.INFO, logger="src.modules.frame_rate_logger"):
            await module.process(Message(frame), AsyncProcessor())

        assert "Processing frame rate" in caplog.text
        assert "FPS" in caplog.text

    asyncio.run(scenario())


def make_test_image() -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :] * 4
    image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None] * 4
    image[:, :, 2] = 80
    return image


def test_image_enhancement_module_enhances_raw_bgr_image() -> None:
    async def scenario() -> None:
        image = make_test_image()
        module = ImageEnhancementModule(
            name="enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )

        routed = await module.process(Message(image), AsyncProcessor())
        enhanced = routed.message.payload

        assert routed.destination == "enhanced_frames"
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == image.shape
        assert enhanced.dtype == image.dtype
        assert not np.array_equal(enhanced, image)
        assert np.array_equal(routed.message.metadata["original_frame_image"], image)
        assert routed.message.metadata["original_frame_image"] is not image

    asyncio.run(scenario())


def test_image_enhancement_module_preserves_video_frame_metadata() -> None:
    async def scenario() -> None:
        image = make_test_image()
        frame = VideoFrame(
            image=image,
            frame_index=42,
            timestamp_seconds=1.25,
            loop_count=3,
        )
        module = ImageEnhancementModule(
            name="enhancer",
            input_queue="frames",
            output_queue="enhanced_frames",
        )

        routed = await module.process(
            Message(frame, metadata={"camera": "test"}), AsyncProcessor()
        )
        enhanced_frame = routed.message.payload

        assert isinstance(enhanced_frame, VideoFrame)
        assert enhanced_frame.frame_index == frame.frame_index
        assert enhanced_frame.timestamp_seconds == frame.timestamp_seconds
        assert enhanced_frame.loop_count == frame.loop_count
        assert enhanced_frame.image.shape == image.shape
        assert enhanced_frame.image.dtype == image.dtype
        assert not np.array_equal(enhanced_frame.image, image)
        assert routed.message.metadata["camera"] == "test"
        assert np.array_equal(routed.message.metadata["original_frame_image"], image)
        assert routed.message.metadata["original_frame_image"] is not image

    asyncio.run(scenario())


def test_image_enhancement_module_routes_output_to_configured_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("frames")
        processor.create_queue("enhanced_frames")
        processor.register_module(
            ImageEnhancementModule(
                name="enhancer",
                input_queue="frames",
                output_queue="enhanced_frames",
            )
        )

        await processor.start()
        await processor.submit("frames", make_test_image())

        result = await asyncio.wait_for(
            processor.queue("enhanced_frames").get(),
            timeout=1,
        )
        assert isinstance(result.payload, np.ndarray)
        assert result.payload.shape == (32, 32, 3)
        processor.queue("enhanced_frames").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_queue_fanout_module_routes_message_to_all_output_queues() -> None:
    async def scenario() -> None:
        module = QueueFanoutModule(
            name="fanout",
            input_queue="in",
            output_queues=["out_a", "out_b"],
        )
        message = Message("payload", metadata={"source": "test"})

        routed = await module.process(message, AsyncProcessor())

        assert [item.destination for item in routed] == ["out_a", "out_b"]
        assert all(item.message is message for item in routed)

    asyncio.run(scenario())


def test_gmm_color_mask_module_outputs_binary_mask(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        write_synthetic_gmm_model(model_path, query_bgr)
        image = np.full((20, 24, 3), query_bgr, dtype=np.uint8)
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
        )

        routed = await module.process(Message(image), AsyncProcessor())

        assert routed.destination == "masks"
        assert routed.message.payload.shape == (20, 24)
        assert routed.message.payload.dtype == np.uint8
        assert set(np.unique(routed.message.payload)).issubset({0, 255})
        assert np.any(routed.message.payload == 255)

    asyncio.run(scenario())


def test_gmm_color_mask_module_writes_debug_mask_when_enabled(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        debug_dir = tmp_path / "debug"
        write_synthetic_gmm_model(model_path, query_bgr)
        image = np.full((20, 24, 3), query_bgr, dtype=np.uint8)
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
            debug=True,
            debug_dir=debug_dir,
        )

        routed = await module.process(Message(image), AsyncProcessor())

        debug_mask_path = debug_dir / "gmm_color_mask.png"
        assert debug_mask_path.exists()
        debug_mask = cv2.imread(str(debug_mask_path), cv2.IMREAD_GRAYSCALE)
        assert debug_mask is not None
        assert debug_mask.shape == routed.message.payload.shape
        assert debug_mask.dtype == np.uint8

    asyncio.run(scenario())


def test_gmm_color_mask_module_preserves_video_frame_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        query_bgr = (10, 200, 20)
        model_path = tmp_path / "color_classifier_gmm.joblib"
        write_synthetic_gmm_model(model_path, query_bgr)
        frame = VideoFrame(
            image=np.full((12, 14, 3), query_bgr, dtype=np.uint8),
            frame_index=42,
            timestamp_seconds=1.68,
            loop_count=3,
        )
        module = GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=model_path,
        )

        routed = await module.process(
            Message(frame, metadata={"source": "test"}), AsyncProcessor()
        )

        assert routed.message.payload.shape == (12, 14)
        assert routed.message.metadata["source"] == "test"
        assert routed.message.metadata["frame_index"] == 42
        assert routed.message.metadata["timestamp_seconds"] == 1.68
        assert routed.message.metadata["loop_count"] == 3

    asyncio.run(scenario())


def test_gmm_color_mask_module_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GMM color classifier model not found"):
        GMMColorMaskModule(
            name="gmm",
            input_queue="frames",
            output_queue="masks",
            model_path=tmp_path / "missing.joblib",
        )


def make_synthetic_marker_image() -> np.ndarray:
    image = np.full((360, 480, 3), 230, dtype=np.uint8)
    marker_quad = np.array(
        [[95, 70], [380, 95], [350, 315], [120, 290]],
        dtype=np.int32,
    )
    inner_quad = np.array(
        [[150, 125], [320, 135], [305, 245], [160, 240]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, marker_quad, (20, 20, 20))
    cv2.fillConvexPoly(image, inner_quad, (240, 240, 240))
    cv2.polylines(image, [marker_quad], True, (0, 0, 0), 8, cv2.LINE_AA)
    return image


def marker_debug_paths(debug_dir: Path, frame_index: int = 0) -> list[Path]:
    return [
        debug_dir / "marker_input.png",
        debug_dir / "marker_hough_lines.png",
        debug_dir / f"marker_detected_quad_{frame_index:04}.png",
        debug_dir / f"marker_rectified_cutout_{frame_index:04}.png",
    ]


def test_marker_candidate_refinement_clips_initial_guess_to_bounds() -> None:
    dist = np.zeros((100, 120), dtype=np.float32)
    initial_quad = np.array(
        [[-6, 10], [126, -3], [118, 108], [4, 96]],
        dtype=np.float32,
    )

    refined = refine_candidate(
        initial_quad,
        dist,
        width=120,
        height=100,
        min_area=400.0,
    )

    assert refined.shape == (4, 2)
    assert np.all(refined[:, 0] >= 0)
    assert np.all(refined[:, 0] <= 119)
    assert np.all(refined[:, 1] >= 0)
    assert np.all(refined[:, 1] <= 99)


def test_marker_bw_dominance_prefers_black_and_white_quad() -> None:
    quad = np.array([[16, 16], [112, 16], [112, 112], [16, 112]], dtype=np.float32)

    bw_image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.fillConvexPoly(bw_image, np.rint(quad).astype(np.int32), (20, 20, 20))

    color_image = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.fillConvexPoly(color_image, np.rint(quad).astype(np.int32), (0, 0, 255))

    bw_score = compute_bw_dominance(bw_image, quad)
    color_score = compute_bw_dominance(color_image, quad)

    assert bw_score > 0.95
    assert color_score < 0.1


def test_marker_nms_prefers_larger_overlapping_bw_quad() -> None:
    large = Candidate(
        quad=np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32),
        source="contour",
        variant_idx=0,
        score=6.0,
        bw_dominance=0.92,
    )
    small = Candidate(
        quad=np.array([[28, 28], [92, 28], [92, 92], [28, 92]], dtype=np.float32),
        source="contour",
        variant_idx=0,
        score=3.0,
        bw_dominance=0.96,
    )
    distant = Candidate(
        quad=np.array([[150, 20], [210, 20], [210, 80], [150, 80]], dtype=np.float32),
        source="contour",
        variant_idx=0,
        score=5.0,
        bw_dominance=0.88,
    )

    kept = apply_nms([small, large, distant], iou_threshold=0.3)

    kept_ids = {id(candidate) for candidate in kept}
    assert id(large) in kept_ids
    assert id(distant) in kept_ids
    assert id(small) not in kept_ids


def test_fit_square_prefers_prior_quad_when_marker_evidence_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    edge_artifacts = EdgeArtifacts(
        gray=np.zeros((120, 120), dtype=np.uint8),
        blur=np.zeros((120, 120), dtype=np.uint8),
        edges_canny=np.zeros((120, 120), dtype=np.uint8),
        grad_mag=np.zeros((120, 120), dtype=np.float32),
        dist=np.zeros((120, 120), dtype=np.float32),
    )
    inner_quad = np.array(
        [[30, 30], [88, 30], [88, 88], [30, 88]],
        dtype=np.float32,
    )
    prior_quad = np.array(
        [[10, 10], [110, 10], [110, 110], [10, 110]],
        dtype=np.float32,
    )

    monkeypatch.setattr(
        marker_rectifier_core,
        "find_contour_candidates",
        lambda *args, **kwargs: [
            Candidate(quad=inner_quad, source="contour", variant_idx=0)
        ],
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "hough_line_debug",
        lambda *args, **kwargs: ([], None),
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "fallback_edge_retry",
        lambda artifacts: artifacts.edges_canny,
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "sample_candidates_by_area",
        lambda candidates: list(candidates),
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "dedupe_candidate_pool",
        lambda candidates: list(candidates),
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "refine_candidate",
        lambda quad, *args, **kwargs: np.asarray(quad, dtype=np.float32),
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "edge_distance_score",
        lambda quad, *args, **kwargs: 1.0 if np.allclose(quad, prior_quad) else 8.0,
    )
    monkeypatch.setattr(
        marker_rectifier_core,
        "compute_bw_dominance",
        lambda image, quad: 0.95 if np.allclose(quad, prior_quad) else 0.75,
    )

    def fake_marker_detection_evidence(
        image: np.ndarray,
        quad: np.ndarray,
        out_size: int = 512,
    ) -> MarkerEvidence:
        del image, out_size
        if np.allclose(quad, inner_quad):
            return MarkerEvidence(detected_count=1, rejected_count=2)
        if np.allclose(quad, prior_quad):
            return MarkerEvidence(detected_count=0, rejected_count=0)
        return MarkerEvidence(detected_count=0, rejected_count=10)

    monkeypatch.setattr(
        marker_rectifier_core,
        "marker_detection_evidence",
        fake_marker_detection_evidence,
    )

    result = marker_rectifier_core.fit_square(
        image,
        [edge_artifacts],
        marker_image=image,
        prior_quad=prior_quad,
    )

    assert np.allclose(result.quad, prior_quad)


def test_marker_rectification_debug_disabled_does_not_create_debug_files(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=False,
            debug_dir=debug_dir,
        )

        routed = await module.process(
            Message(make_synthetic_marker_image()), AsyncProcessor()
        )

        assert routed is not None
        assert not debug_dir.exists()

    asyncio.run(scenario())


def test_marker_rectification_uses_trusted_prior_without_full_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        image = make_synthetic_marker_image()
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )
        prior_quad = np.array(
            [
                [0, 0],
                [image.shape[1] - 1, 0],
                [image.shape[1] - 1, image.shape[0] - 1],
                [0, image.shape[0] - 1],
            ],
            dtype=np.float32,
        )

        def fail_build_edge_variants(*args: object, **kwargs: object) -> object:
            raise AssertionError("trusted prior should bypass full rectifier search")

        monkeypatch.setattr(
            marker_rectifier_module,
            "build_edge_variants",
            fail_build_edge_variants,
        )

        routed = await module.process(
            Message(image, metadata={"prior_source_quad": prior_quad.tolist()}),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.message.metadata["rectifier_search_mode"] == "trusted_prior"
        assert routed.message.metadata["score"] == 0.0
        assert np.allclose(
            np.asarray(routed.message.metadata["source_quad"], dtype=np.float32),
            prior_quad,
        )

    asyncio.run(scenario())


def test_marker_rectification_debug_enabled_writes_processing_images(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )

        routed = await module.process(
            Message(make_synthetic_marker_image()), AsyncProcessor()
        )

        assert routed is not None
        for debug_path in marker_debug_paths(debug_dir):
            assert debug_path.exists()
            assert cv2.imread(str(debug_path)) is not None
        assert cv2.imread(str(debug_dir / "marker_input.png")).shape == (360, 480, 3)
        assert cv2.imread(
            str(debug_dir / "marker_rectified_cutout_0000.png")
        ).shape == (
            512,
            512,
            3,
        )

    asyncio.run(scenario())


def test_marker_rectification_debug_enabled_writes_failure_images(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )
        blank = np.full((240, 320, 3), 127, dtype=np.uint8)

        routed = await module.process(Message(blank), AsyncProcessor())

        assert routed is None
        for debug_path in marker_debug_paths(debug_dir):
            assert debug_path.exists()
            assert cv2.imread(str(debug_path)) is not None
        cutout = cv2.imread(str(debug_dir / "marker_rectified_cutout_0000.png"))
        assert cutout.shape == (512, 512, 3)
        assert int(np.count_nonzero(cutout)) == 0

    asyncio.run(scenario())


def test_marker_rectification_debug_detected_quad_uses_frame_index(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )
        frame = VideoFrame(
            image=make_synthetic_marker_image(),
            frame_index=12,
            timestamp_seconds=0.48,
            loop_count=0,
        )

        routed = await module.process(Message(frame), AsyncProcessor())

        assert routed is not None
        assert (debug_dir / "marker_detected_quad_0012.png").exists()
        assert (debug_dir / "marker_rectified_cutout_0012.png").exists()
        assert not (debug_dir / "marker_detected_quad_0000.png").exists()
        assert not (debug_dir / "marker_rectified_cutout_0000.png").exists()

    asyncio.run(scenario())


def test_marker_rectification_debug_detected_quad_increments_for_raw_arrays(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
            debug=True,
            debug_dir=debug_dir,
        )

        routed0 = await module.process(
            Message(make_synthetic_marker_image()), AsyncProcessor()
        )
        routed1 = await module.process(
            Message(make_synthetic_marker_image()), AsyncProcessor()
        )

        assert routed0 is not None
        assert routed1 is not None
        assert (debug_dir / "marker_detected_quad_0000.png").exists()
        assert (debug_dir / "marker_detected_quad_0001.png").exists()
        assert (debug_dir / "marker_rectified_cutout_0000.png").exists()
        assert (debug_dir / "marker_rectified_cutout_0001.png").exists()

    asyncio.run(scenario())


def test_marker_rectification_uses_enhanced_payload_when_original_metadata_present() -> (
    None
):
    async def scenario() -> None:
        original = make_synthetic_marker_image()
        enhanced = apply_enhancement(original, "underwater")
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )
        frame = VideoFrame(
            image=enhanced,
            frame_index=0,
            timestamp_seconds=0.0,
            loop_count=0,
        )

        routed = await module.process(
            Message(frame, metadata={ORIGINAL_FRAME_METADATA_KEY: original.copy()}),
            AsyncProcessor(),
        )

        assert routed is not None
        assert np.array_equal(routed.message.metadata["source_frame_image"], enhanced)

    asyncio.run(scenario())


def test_marker_rectification_matches_video_1_frame_0_ground_truth() -> None:
    async def scenario() -> None:
        image = cv2.imread("data/aruco/video-1/frame_0001.jpg")
        assert image is not None

        with open("data/video-1-marker-gt.json", encoding="utf-8") as handle:
            gt = np.asarray(json.load(handle)["frames"][0]["markers"], dtype=np.float32)

        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )
        frame = VideoFrame(
            image=image,
            frame_index=0,
            timestamp_seconds=0.0,
            loop_count=0,
        )

        routed = await module.process(Message(frame), AsyncProcessor())

        assert routed is not None
        quad = np.asarray(routed.message.metadata["quad"], dtype=np.float32)
        assert quad_iou(quad, gt) >= 0.85

    asyncio.run(scenario())


def test_marker_rectification_matches_ground_truth_with_enhanced_payload() -> None:
    def expected_min_iou(frame_index: int) -> float:
        if frame_index in {121, 122}:
            return 0.80
        return 0.85

    async def scenario() -> None:
        with open("data/video-1-marker-gt.json", encoding="utf-8") as handle:
            labeled_frames = json.load(handle)["frames"]

        for labeled in labeled_frames:
            frame_index = int(labeled["frame"])
            original = cv2.imread(f"data/aruco/video-1/frame_{frame_index + 1:04d}.jpg")
            assert original is not None
            enhanced = apply_enhancement(original, "underwater")
            gt = np.asarray(labeled["markers"], dtype=np.float32)

            module = MarkerRectificationModule(
                name="rectifier",
                input_queue="frames",
                output_queue="cutouts",
            )
            frame = VideoFrame(
                image=enhanced,
                frame_index=frame_index,
                timestamp_seconds=0.0,
                loop_count=0,
            )

            routed = await module.process(
                Message(frame, metadata={ORIGINAL_FRAME_METADATA_KEY: original.copy()}),
                AsyncProcessor(),
            )

            assert routed is not None
            quad = np.asarray(routed.message.metadata["quad"], dtype=np.float32)
        assert quad_iou(quad, gt) >= expected_min_iou(frame_index), frame_index

    asyncio.run(scenario())


def test_main_fans_out_enhanced_frames_when_gmm_model_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class SpyOpticalFlowMarkerTrackingModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["tracker_input_queue"] = input_queue
            captured["tracker_output_queue"] = output_queue
            captured["tracker_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyArucoMarkerAnnotationModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["annotator_input_queue"] = input_queue
            captured["annotator_output_queue"] = output_queue
            captured["annotator_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyQueueFanoutModule(BaseModule[np.ndarray]):
        def __init__(
            self, name: str, input_queue: str, output_queues: list[str]
        ) -> None:
            super().__init__(name, input_queue)
            captured["fanout_input_queue"] = input_queue
            captured["fanout_output_queues"] = output_queues

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    missing_model_path = tmp_path / "missing.joblib"
    monkeypatch.setattr(app_main, "GMM_MODEL_PATH", missing_model_path)
    monkeypatch.setattr(
        app_main, "OpticalFlowMarkerTrackingModule", SpyOpticalFlowMarkerTrackingModule
    )
    monkeypatch.setattr(
        app_main, "ArucoMarkerAnnotationModule", SpyArucoMarkerAnnotationModule
    )
    monkeypatch.setattr(app_main, "QueueFanoutModule", SpyQueueFanoutModule)
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--input-path", str(TEST_VIDEO_PATH)])

    asyncio.run(app_main.run_app(args))

    assert captured["fanout_input_queue"] == app_main.ENHANCED_FRAME_QUEUE
    assert captured["fanout_output_queues"] == [app_main.MARKER_FRAME_QUEUE]
    assert captured["tracker_input_queue"] == app_main.MARKER_FRAME_QUEUE
    assert captured["tracker_output_queue"] == app_main.ARUCO_DETECTIONS_QUEUE
    assert captured["annotator_input_queue"] == app_main.ARUCO_DETECTIONS_QUEUE
    assert captured["annotator_output_queue"] == app_main.ANNOTATED_FRAMES_QUEUE
    assert captured["annotator_kwargs"] == {
        "debug": False,
        "debug_dir": Path("data/debug"),
    }


def test_main_registers_gmm_fanout_path_when_model_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class SpyOpticalFlowMarkerTrackingModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["tracker_input_queue"] = input_queue
            captured["tracker_output_queue"] = output_queue
            captured["tracker_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyGMMColorMaskModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["gmm_input_queue"] = input_queue
            captured["gmm_output_queue"] = output_queue
            captured["gmm_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyArucoMarkerAnnotationModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["annotator_input_queue"] = input_queue
            captured["annotator_output_queue"] = output_queue
            captured["annotator_kwargs"] = kwargs

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class SpyQueueFanoutModule(BaseModule[np.ndarray]):
        def __init__(
            self, name: str, input_queue: str, output_queues: list[str]
        ) -> None:
            super().__init__(name, input_queue)
            captured["fanout_input_queue"] = input_queue
            captured["fanout_output_queues"] = output_queues

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    model_path = tmp_path / "color_classifier_gmm.joblib"
    model_path.write_bytes(b"exists")
    monkeypatch.setattr(app_main, "GMM_MODEL_PATH", model_path)
    monkeypatch.setattr(
        app_main, "OpticalFlowMarkerTrackingModule", SpyOpticalFlowMarkerTrackingModule
    )
    monkeypatch.setattr(app_main, "GMMColorMaskModule", SpyGMMColorMaskModule)
    monkeypatch.setattr(
        app_main, "ArucoMarkerAnnotationModule", SpyArucoMarkerAnnotationModule
    )
    monkeypatch.setattr(app_main, "QueueFanoutModule", SpyQueueFanoutModule)
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--input-path", str(TEST_VIDEO_PATH)])

    asyncio.run(app_main.run_app(args))

    assert captured["fanout_input_queue"] == app_main.ENHANCED_FRAME_QUEUE
    assert captured["fanout_output_queues"] == [
        app_main.MARKER_FRAME_QUEUE,
        app_main.GMM_FRAME_QUEUE,
    ]
    assert captured["tracker_input_queue"] == app_main.MARKER_FRAME_QUEUE
    assert captured["tracker_output_queue"] == app_main.ARUCO_DETECTIONS_QUEUE
    assert captured["annotator_input_queue"] == app_main.ARUCO_DETECTIONS_QUEUE
    assert captured["annotator_output_queue"] == app_main.ANNOTATED_FRAMES_QUEUE
    assert captured["annotator_kwargs"] == {
        "debug": False,
        "debug_dir": Path("data/debug"),
    }
    assert captured["gmm_input_queue"] == app_main.GMM_FRAME_QUEUE
    assert captured["gmm_output_queue"] == app_main.COLOR_MASK_QUEUE
    assert captured["gmm_kwargs"] == {
        "model_path": model_path,
        "debug": False,
        "debug_dir": Path("data/debug"),
    }


def test_main_debug_flag_is_parsed_and_wired_to_optical_flow_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class SpyOpticalFlowMarkerTrackingModule(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_queue: str,
            **kwargs: object,
        ) -> None:
            super().__init__(name, input_queue)
            captured["output_queue"] = output_queue
            captured.update(kwargs)

        async def process(
            self,
            message: Message[np.ndarray],
            context: ModuleContext,
        ) -> None:
            return None

    class NoopProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run_until_interrupted(self) -> None:
            return None

    monkeypatch.setattr(
        app_main, "OpticalFlowMarkerTrackingModule", SpyOpticalFlowMarkerTrackingModule
    )
    monkeypatch.setattr(app_main, "ProcessorLoop", NoopProcessorLoop)
    args = app_main.parse_args(["--debug", "--input-path", str(TEST_VIDEO_PATH)])

    assert args.debug is True
    asyncio.run(app_main.run_app(args))

    assert captured["output_queue"] == app_main.ARUCO_DETECTIONS_QUEUE
    assert captured["debug"] is True
    assert captured["debug_dir"] == Path("data/debug")


def make_aruco_detection_result(
    detections: list[ArucoMarkerDetection],
) -> ArucoDetectionResult:
    return ArucoDetectionResult(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        detections=detections,
        rejected_candidates=[],
    )


def test_aruco_marker_annotation_module_maps_cutout_center_to_raw_frame() -> None:
    async def scenario() -> None:
        raw = np.zeros((160, 220, 3), dtype=np.uint8)
        corners = np.array(
            [[8, 18], [12, 18], [12, 22], [8, 22]],
            dtype=np.float32,
        )
        result = make_aruco_detection_result(
            [ArucoMarkerDetection(marker_id=42, corners=corners)]
        )
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
        )
        captured: dict[str, object] = {}

        def fake_draw_marker_id(
            image: np.ndarray, marker_id: int, center: np.ndarray
        ) -> None:
            captured["marker_id"] = marker_id
            captured["center"] = center.copy()
            cv2.circle(
                image, tuple(np.rint(center).astype(np.int32)), 2, (255, 255, 255), -1
            )

        module._draw_marker_id = fake_draw_marker_id  # type: ignore[method-assign]
        routed = await module.process(
            Message(
                result,
                metadata={
                    "original_frame_image": raw,
                    "cutout_to_source_homography": [
                        [1.0, 0.0, 100.0],
                        [0.0, 1.0, 50.0],
                        [0.0, 0.0, 1.0],
                    ],
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.destination == "annotated"
        assert captured["marker_id"] == 42
        assert np.allclose(
            captured["center"], np.array([110.0, 70.0], dtype=np.float32)
        )
        assert np.any(routed.message.payload[68:73, 108:113] != 0)

    asyncio.run(scenario())


def test_aruco_marker_annotation_module_draws_on_copy_and_preserves_metadata() -> None:
    async def scenario() -> None:
        raw = np.full((140, 180, 3), 30, dtype=np.uint8)
        original = raw.copy()
        detections = [
            ArucoMarkerDetection(
                marker_id=7,
                corners=np.array(
                    [[35, 35], [45, 35], [45, 45], [35, 45]], dtype=np.float32
                ),
            ),
            ArucoMarkerDetection(
                marker_id=8,
                corners=np.array(
                    [[75, 55], [85, 55], [85, 65], [75, 65]], dtype=np.float32
                ),
            ),
        ]
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
        )

        routed = await module.process(
            Message(
                make_aruco_detection_result(detections),
                metadata={
                    "original_frame_image": raw,
                    "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                    "frame_index": 4,
                    "timestamp_seconds": 1.25,
                    "loop_count": 2,
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert np.array_equal(raw, original)
        assert not np.array_equal(routed.message.payload, raw)
        assert routed.message.metadata["frame_index"] == 4
        assert routed.message.metadata["timestamp_seconds"] == 1.25
        assert routed.message.metadata["loop_count"] == 2
        assert routed.message.metadata["annotated_marker_count"] == 2
        assert routed.message.metadata["annotated_marker_ids"] == [7, 8]

    asyncio.run(scenario())


def test_aruco_marker_annotation_module_falls_back_to_rectifier_source_frame() -> None:
    async def scenario() -> None:
        source = np.full((120, 120, 3), 20, dtype=np.uint8)
        detection = ArucoMarkerDetection(
            marker_id=5,
            corners=np.array(
                [[45, 45], [55, 45], [55, 55], [45, 55]], dtype=np.float32
            ),
        )
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
        )

        routed = await module.process(
            Message(
                make_aruco_detection_result([detection]),
                metadata={
                    "source_frame_image": source,
                    "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert not np.array_equal(routed.message.payload, source)

    asyncio.run(scenario())


def test_aruco_marker_annotation_module_writes_debug_image(tmp_path: Path) -> None:
    async def scenario() -> None:
        debug_dir = tmp_path / "debug"
        raw = np.full((120, 120, 3), 30, dtype=np.uint8)
        detection = ArucoMarkerDetection(
            marker_id=9,
            corners=np.array(
                [[45, 45], [55, 45], [55, 55], [45, 55]], dtype=np.float32
            ),
        )
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
            debug=True,
            debug_dir=debug_dir,
        )

        routed = await module.process(
            Message(
                make_aruco_detection_result([detection]),
                metadata={
                    "original_frame_image": raw,
                    "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                    "frame_index": 12,
                    "timestamp_seconds": 3.456,
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        path = debug_dir / "aruco_annotated_frame.png"
        assert path.exists()
        debug_image = cv2.imread(str(path))
        assert debug_image is not None
        lower_left = debug_image[-44:-8, 8:40]
        assert np.any(lower_left != raw[-44:-8, 8:40])
        assert np.array_equal(routed.message.payload[-44:-8, 8:40], raw[-44:-8, 8:40])

    asyncio.run(scenario())


def aruco_detection(
    marker_id: int, x: float, y: float, size: float = 10.0
) -> ArucoMarkerDetection:
    return ArucoMarkerDetection(
        marker_id=marker_id,
        corners=np.array(
            [
                [x, y],
                [x + size, y],
                [x + size, y + size],
                [x, y + size],
            ],
            dtype=np.float32,
        ),
    )


def write_marker_template(
    template_dir: Path,
    marker_id: int,
    color: tuple[int, int, int],
    *,
    size: int = 12,
) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(template_dir / f"6x6_1000_{marker_id:04d}.png"), image)


def test_aruco_marker_annotation_module_loads_marker_template_by_id(
    tmp_path: Path,
) -> None:
    template_dir = tmp_path / "6x6_1000"
    write_marker_template(template_dir, 7, (10, 20, 30), size=8)
    module = ArucoMarkerAnnotationModule(
        name="annotator",
        input_queue="detections",
        output_queue="annotated",
        marker_template_dir=template_dir,
        template_marker_size=16,
    )

    template = module._load_marker_template(7, 16)

    assert template.shape == (16, 16, 3)
    assert np.array_equal(template[8, 8], np.array([10, 20, 30], dtype=np.uint8))


def test_aruco_marker_annotation_module_infers_grid_from_shuffled_detections(
    tmp_path: Path,
) -> None:
    module = ArucoMarkerAnnotationModule(
        name="annotator",
        input_queue="detections",
        output_queue="annotated",
        marker_template_dir=tmp_path / "6x6_1000",
    )
    detections = [
        aruco_detection(5, 50, 40),
        aruco_detection(2, 20, 10),
        aruco_detection(6, 80, 40),
        aruco_detection(1, 50, 10),
        aruco_detection(4, 20, 40),
        aruco_detection(3, 80, 10),
    ]

    grid = module._infer_marker_grid(detections)

    assert grid == [[2, 1, 3], [4, 5, 6]]


def test_aruco_marker_annotation_module_draws_template_grid_in_lower_right(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        template_dir = tmp_path / "6x6_1000"
        write_marker_template(template_dir, 1, (0, 255, 0))
        raw = np.zeros((120, 160, 3), dtype=np.uint8)
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
            marker_template_dir=template_dir,
            template_marker_size=20,
            template_margin_pixels=5,
        )

        routed = await module.process(
            Message(
                make_aruco_detection_result([aruco_detection(1, 10, 10)]),
                metadata={
                    "original_frame_image": raw,
                    "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.message.metadata["template_grid_shape"] == (1, 1)
        grid_region = routed.message.payload[95:115, 135:155]
        assert np.array_equal(
            grid_region[10, 10], np.array([0, 255, 0], dtype=np.uint8)
        )

    asyncio.run(scenario())


def test_aruco_marker_annotation_module_shrinks_template_grid_to_fit_frame(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        template_dir = tmp_path / "6x6_1000"
        for marker_id in range(1, 5):
            write_marker_template(
                template_dir, marker_id, (marker_id, marker_id, marker_id)
            )
        raw = np.zeros((50, 50, 3), dtype=np.uint8)
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
            marker_template_dir=template_dir,
            template_marker_size=40,
            template_margin_pixels=5,
        )
        detections = [
            aruco_detection(1, 10, 10),
            aruco_detection(2, 30, 10),
            aruco_detection(3, 10, 30),
            aruco_detection(4, 30, 30),
        ]

        routed = await module.process(
            Message(
                make_aruco_detection_result(detections),
                metadata={
                    "original_frame_image": raw,
                    "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
                },
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.message.metadata["template_grid_shape"] == (2, 2)
        assert np.array_equal(
            routed.message.payload[6, 6], np.array([1, 1, 1], dtype=np.uint8)
        )
        assert np.array_equal(
            routed.message.payload[6, 26], np.array([2, 2, 2], dtype=np.uint8)
        )
        assert np.array_equal(
            routed.message.payload[26, 6], np.array([3, 3, 3], dtype=np.uint8)
        )
        assert np.array_equal(
            routed.message.payload[26, 26], np.array([4, 4, 4], dtype=np.uint8)
        )

    asyncio.run(scenario())


def test_aruco_marker_annotation_module_uses_placeholder_for_missing_template(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        template_dir = tmp_path / "6x6_1000"
        template_dir.mkdir()
        raw = np.zeros((90, 90, 3), dtype=np.uint8)
        module = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
            marker_template_dir=template_dir,
            template_marker_size=20,
            template_margin_pixels=5,
        )

        with caplog.at_level(
            logging.WARNING, logger="src.modules.aruco_marker_annotator"
        ):
            routed = await module.process(
                Message(
                    make_aruco_detection_result([aruco_detection(999, 10, 10)]),
                    metadata={
                        "original_frame_image": raw,
                        "cutout_to_source_homography": np.eye(
                            3, dtype=np.float32
                        ).tolist(),
                    },
                ),
                AsyncProcessor(),
            )

        assert routed is not None
        assert "Missing or unreadable ArUco marker template" in caplog.text
        grid_region = routed.message.payload[65:85, 65:85]
        assert np.all(grid_region == 255)

    asyncio.run(scenario())


class FakeTrackerRectifier(BaseModule[np.ndarray]):
    def __init__(self) -> None:
        super().__init__("fake-rectifier", "frames")
        self.calls = 0

    async def process(
        self,
        message: Message[np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray]:
        self.calls += 1
        image = (
            message.payload.image
            if isinstance(message.payload, (ImageFrame, VideoFrame))
            else message.payload
        )
        metadata = dict(message.metadata)
        metadata.update(
            {
                "source_frame_image": image.copy(),
                "source_quad": np.array(
                    [
                        [0, 0],
                        [image.shape[1] - 1, 0],
                        [image.shape[1] - 1, image.shape[0] - 1],
                        [0, image.shape[0] - 1],
                    ],
                    dtype=np.float32,
                ).tolist(),
                "cutout_to_source_homography": np.eye(3, dtype=np.float32).tolist(),
            }
        )
        return RoutedMessage("cutouts", Message(image, metadata=metadata))


class CapturingTrackerRectifier(FakeTrackerRectifier):
    def __init__(self) -> None:
        super().__init__()
        self.seen_metadata: list[dict[str, object]] = []

    async def process(
        self,
        message: Message[np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[np.ndarray]:
        self.seen_metadata.append(dict(message.metadata))
        return await super().process(message, context)


class FakeTrackerDetector(BaseModule[np.ndarray]):
    def __init__(self, detections: list[ArucoMarkerDetection]) -> None:
        super().__init__("fake-detector", "cutouts")
        self.detections = detections
        self.calls = 0

    async def process(
        self,
        message: Message[np.ndarray],
        context: ModuleContext,
    ) -> RoutedMessage[ArucoDetectionResult] | None:
        self.calls += 1
        if not self.detections:
            return None
        metadata = dict(message.metadata)
        metadata.update(
            {
                "dictionary_name": "DICT_6X6_1000",
                "marker_count": len(self.detections),
                "marker_ids": [detection.marker_id for detection in self.detections],
                "detection_passes": {
                    detection.marker_id: "fake" for detection in self.detections
                },
            }
        )
        return RoutedMessage(
            "detections",
            Message(
                ArucoDetectionResult(
                    image=message.payload,
                    detections=self.detections,
                    rejected_candidates=[],
                ),
                metadata=metadata,
            ),
        )


def marker_corner_image(
    detections: list[ArucoMarkerDetection],
    *,
    shape: tuple[int, int, int] = (140, 180, 3),
) -> np.ndarray:
    image = np.full(shape, 255, dtype=np.uint8)
    for detection in detections:
        for corner in np.rint(detection.corners).astype(np.int32):
            cv2.circle(
                image,
                tuple(int(value) for value in corner),
                4,
                (0, 0, 0),
                -1,
                cv2.LINE_AA,
            )
    return image


def translated_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255)
    )


def test_optical_flow_marker_tracker_fallback_seeds_state() -> None:
    async def scenario() -> None:
        detection = aruco_detection(10, 30, 30, 20)
        rectifier = FakeTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )

        routed = await module.process(
            Message(marker_corner_image([detection])), AsyncProcessor()
        )

        assert routed is not None
        assert routed.destination == "detections"
        assert rectifier.calls == 1
        assert detector.calls == 1
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["detection_coordinate_space"] == "cutout"
        assert module._state is not None

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_predicts_next_frame_without_fallback() -> None:
    async def scenario() -> None:
        detection = aruco_detection(11, 35, 35, 22)
        rectifier = CapturingTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )
        first = marker_corner_image([detection])
        second = translated_image(first, 6, 4)

        class FakeTrackResult:
            def __init__(
                self,
                corners: np.ndarray | None,
                confidence: float | None,
                reason: str,
            ) -> None:
                self.corners = corners
                self.confidence = confidence
                self.reason = reason
                self.succeeded = corners is not None and confidence is not None

        await module.process(Message(first), AsyncProcessor())

        def fake_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FakeTrackResult:
            del previous_gray, gray, image_shape
            points_arr = np.asarray(points, dtype=np.float32)
            return FakeTrackResult(
                points_arr + np.array([6, 4], dtype=np.float32),
                0.9,
                "tracked",
            )

        module._track_quad_result = fake_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(Message(second), AsyncProcessor())

        assert routed is not None
        assert rectifier.calls == 2
        assert detector.calls == 2
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["tracking_refresh_reason"] == "trusted_quad"
        assert routed.message.metadata["detection_coordinate_space"] == "cutout"
        assert len(rectifier.seen_metadata) == 2
        assert "prior_source_quad" in rectifier.seen_metadata[1]
        expected_quad = np.array(
            [
                [6, 4],
                [first.shape[1] - 1 + 6, 4],
                [first.shape[1] - 1 + 6, first.shape[0] - 1 + 4],
                [6, first.shape[0] - 1 + 4],
            ],
            dtype=np.float32,
        )
        assert np.allclose(
            np.asarray(
                rectifier.seen_metadata[1]["prior_source_quad"], dtype=np.float32
            ),
            expected_quad,
        )
        assert "force_full_rectifier" not in rectifier.seen_metadata[1]

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_writes_frame_indexed_prediction_debug_image(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        detection = aruco_detection(12, 35, 35, 22)
        rectifier = FakeTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
            debug=True,
            debug_dir=tmp_path,
        )
        first = marker_corner_image([detection])
        second = translated_image(first, 6, 4)

        class FakeTrackResult:
            def __init__(
                self,
                corners: np.ndarray | None,
                confidence: float | None,
                reason: str,
            ) -> None:
                self.corners = corners
                self.confidence = confidence
                self.reason = reason
                self.succeeded = corners is not None and confidence is not None

        await module.process(
            Message(
                VideoFrame(first, frame_index=5, timestamp_seconds=0.2, loop_count=0),
            ),
            AsyncProcessor(),
        )

        def fake_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FakeTrackResult:
            del previous_gray, gray, image_shape
            points_arr = np.asarray(points, dtype=np.float32)
            return FakeTrackResult(
                points_arr + np.array([6, 4], dtype=np.float32),
                0.9,
                "tracked",
            )

        module._track_quad_result = fake_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(
            Message(
                VideoFrame(second, frame_index=6, timestamp_seconds=0.24, loop_count=0),
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["tracking_refresh_reason"] == "trusted_quad"
        fallback_path = tmp_path / "optical_flow_corners_frame_000005.png"
        prediction_path = tmp_path / "optical_flow_corners_frame_000006.png"
        tracked_quad_path = tmp_path / "marker_detected_quad_0006.png"
        assert fallback_path.exists()
        assert prediction_path.exists()
        assert tracked_quad_path.exists()
        fallback_image = cv2.imread(str(fallback_path))
        assert fallback_image is not None
        assert fallback_image.shape == first.shape
        assert not np.array_equal(fallback_image, first)
        debug_image = cv2.imread(str(prediction_path))
        assert debug_image is not None
        assert debug_image.shape == second.shape
        assert not np.array_equal(debug_image, second)
        tracked_quad_image = cv2.imread(str(tracked_quad_path))
        assert tracked_quad_image is not None
        assert tracked_quad_image.shape == second.shape
        assert not np.array_equal(tracked_quad_image, second)

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_writes_failed_prediction_debug_image(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        detection = aruco_detection(13, 35, 35, 22)
        rectifier = FakeTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
            debug=True,
            debug_dir=tmp_path,
        )
        first = marker_corner_image([detection])
        second = np.full(first.shape, 255, dtype=np.uint8)

        await module.process(
            Message(
                VideoFrame(first, frame_index=5, timestamp_seconds=0.2, loop_count=0)
            ),
            AsyncProcessor(),
        )

        class FailedTrackResult:
            corners = None
            confidence = None
            reason = "too_few_valid_points"
            succeeded = False

        def fail_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FailedTrackResult:
            return FailedTrackResult()

        module._track_quad_result = fail_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(
            Message(
                VideoFrame(second, frame_index=6, timestamp_seconds=0.24, loop_count=0)
            ),
            AsyncProcessor(),
        )

        assert routed is not None
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["tracking_refresh_reason"] == "quad_track_failed"
        failure_path = tmp_path / "optical_flow_corners_frame_000006.png"
        assert failure_path.exists()
        debug_image = cv2.imread(str(failure_path))
        assert debug_image is not None
        assert debug_image.shape == second.shape
        assert not np.array_equal(debug_image, second)
        assert rectifier.calls == 2
        assert detector.calls == 2

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_preserves_last_quad_when_quad_tracking_fails() -> (
    None
):
    async def scenario() -> None:
        detection = aruco_detection(15, 35, 35, 22)
        rectifier = FakeTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )
        first = marker_corner_image([detection])
        second = translated_image(first, 4, 3)

        await module.process(Message(first), AsyncProcessor())

        assert module._state is not None
        previous_quad = np.asarray(module._state.quad, dtype=np.float32)

        class FakeTrackResult:
            def __init__(
                self,
                corners: np.ndarray | None,
                confidence: float | None,
                reason: str,
            ) -> None:
                self.corners = corners
                self.confidence = confidence
                self.reason = reason
                self.succeeded = corners is not None and confidence is not None

        def fake_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FakeTrackResult:
            del previous_gray, gray, image_shape
            points_arr = np.asarray(points, dtype=np.float32)
            if float(np.max(points_arr)) > 100.0:
                return FakeTrackResult(None, None, "too_few_valid_points")
            return FakeTrackResult(
                points_arr + np.array([4, 3], dtype=np.float32),
                0.9,
                "tracked",
            )

        module._track_quad_result = fake_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(Message(second), AsyncProcessor())

        assert routed is not None
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["tracking_refresh_reason"] == "quad_track_failed"
        assert module._state is not None
        assert np.allclose(module._state.quad, previous_quad)

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_passes_prior_quad_to_fallback_after_failure() -> (
    None
):
    async def scenario() -> None:
        detection = aruco_detection(14, 35, 35, 22)
        rectifier = CapturingTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )
        first = marker_corner_image([detection])
        second = np.full(first.shape, 255, dtype=np.uint8)

        await module.process(Message(first), AsyncProcessor())

        class FailedTrackResult:
            corners = None
            confidence = None
            reason = "too_few_valid_points"
            succeeded = False

        def fail_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FailedTrackResult:
            return FailedTrackResult()

        module._track_quad_result = fail_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(Message(second), AsyncProcessor())

        assert routed is not None
        assert len(rectifier.seen_metadata) == 2
        assert "prior_source_quad" in rectifier.seen_metadata[1]
        assert rectifier.seen_metadata[1]["force_full_rectifier"] is True
        prior_quad = np.asarray(
            rectifier.seen_metadata[1]["prior_source_quad"],
            dtype=np.float32,
        )
        expected_quad = np.array(
            [
                [0, 0],
                [first.shape[1] - 1, 0],
                [first.shape[1] - 1, first.shape[0] - 1],
                [0, first.shape[0] - 1],
            ],
            dtype=np.float32,
        )
        assert np.allclose(prior_quad, expected_quad)

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_uses_support_points_when_exact_corners_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corners = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.float32)
    module = OpticalFlowMarkerTrackingModule(
        name="tracker",
        input_queue="frames",
        output_queue="detections",
    )
    previous = np.zeros((100, 100), dtype=np.uint8)
    current = np.zeros((100, 100), dtype=np.uint8)
    delta = np.array([5, 3], dtype=np.float32)
    calls = {"count": 0}

    def fake_lk(
        previous_gray: np.ndarray,
        gray: np.ndarray,
        previous_points: np.ndarray,
        next_points: np.ndarray | None,
        *args: object,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls["count"] += 1
        points = previous_points.reshape(-1, 2).astype(np.float32)
        status = np.ones((len(points), 1), dtype=np.uint8)
        errors = np.ones((len(points), 1), dtype=np.float32)
        if calls["count"] % 2 == 1:
            for idx, point in enumerate(points):
                if np.any(np.all(np.isclose(corners, point, atol=1e-5), axis=1)):
                    status[idx, 0] = 0
            return (points + delta).reshape(-1, 1, 2), status, errors
        return (points - delta).reshape(-1, 1, 2), status, errors

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)

    tracked, confidence = module._track_quad(previous, current, corners, (100, 100, 3))

    assert tracked is not None
    assert confidence is not None
    assert np.allclose(tracked, corners + delta, atol=0.25)
    assert confidence > 0.5


def test_optical_flow_marker_tracker_forces_full_rectifier_on_low_quad_confidence() -> (
    None
):
    async def scenario() -> None:
        detection = aruco_detection(20, 25, 45, 20)
        rectifier = CapturingTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )
        first = marker_corner_image([detection])
        second = translated_image(first, 12, 0)

        await module.process(Message(first), AsyncProcessor())

        class FakeTrackResult:
            def __init__(
                self,
                corners: np.ndarray | None,
                confidence: float | None,
                reason: str,
            ) -> None:
                self.corners = corners
                self.confidence = confidence
                self.reason = reason
                self.succeeded = corners is not None and confidence is not None

        def fake_track_quad_result(
            previous_gray: np.ndarray,
            gray: np.ndarray,
            points: np.ndarray,
            image_shape: tuple[int, ...],
        ) -> FakeTrackResult:
            del previous_gray, gray, image_shape
            return FakeTrackResult(
                points + np.array([12, 0], dtype=np.float32), 0.2, "tracked"
            )

        module._track_quad_result = fake_track_quad_result  # type: ignore[method-assign]
        routed = await module.process(Message(second), AsyncProcessor())

        assert routed is not None
        assert routed.message.metadata["tracking_source"] == "detector"
        assert routed.message.metadata["tracking_refresh_reason"] == "low_confidence"
        assert rectifier.calls == 2
        assert detector.calls == 2
        assert rectifier.seen_metadata[1]["force_full_rectifier"] is True
        assert "prior_source_quad" in rectifier.seen_metadata[1]

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_reruns_detector_when_tracking_fails() -> None:
    async def scenario() -> None:
        detection = aruco_detection(30, 40, 40, 20)
        rectifier = FakeTrackerRectifier()
        detector = FakeTrackerDetector([detection])
        module = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=rectifier,
            detector=detector,
        )

        await module.process(
            Message(marker_corner_image([detection])), AsyncProcessor()
        )
        routed = await module.process(
            Message(np.full((140, 180, 3), 255, dtype=np.uint8)), AsyncProcessor()
        )

        assert routed is not None
        assert routed.message.metadata["tracking_source"] == "detector"
        assert rectifier.calls == 2
        assert detector.calls == 2

    asyncio.run(scenario())


def test_optical_flow_marker_tracker_fallback_output_is_annotator_compatible() -> None:
    async def scenario() -> None:
        raw = np.full((120, 160, 3), 20, dtype=np.uint8)
        detection = aruco_detection(40, 50, 50, 20)
        tracker = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            rectifier=FakeTrackerRectifier(),
            detector=FakeTrackerDetector([detection]),
        )
        annotator = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
        )

        tracked = await tracker.process(Message(raw), AsyncProcessor())
        assert tracked is not None
        annotated = await annotator.process(tracked.message, AsyncProcessor())

        assert annotated is not None
        assert annotated.message.metadata["annotated_marker_ids"] == [40]
        assert not np.array_equal(annotated.message.payload, raw)

    asyncio.run(scenario())


def test_marker_rectification_module_outputs_rectified_cutout() -> None:
    async def scenario() -> None:
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )

        routed = await module.process(
            Message(make_synthetic_marker_image()), AsyncProcessor()
        )

        assert routed is not None
        assert routed.destination == "cutouts"
        assert routed.message.payload.shape == (512, 512, 3)
        assert routed.message.payload.dtype == np.uint8
        assert "quad" in routed.message.metadata
        assert "source_quad" in routed.message.metadata
        assert "score" in routed.message.metadata
        assert routed.message.metadata["input_shape"] == (360, 480, 3)
        assert routed.message.metadata["cutout_size"] == 512
        assert np.asarray(
            routed.message.metadata["source_to_cutout_homography"]
        ).shape == (3, 3)
        assert np.asarray(
            routed.message.metadata["cutout_to_source_homography"]
        ).shape == (3, 3)
        assert np.array_equal(
            routed.message.metadata["source_frame_image"], make_synthetic_marker_image()
        )

    asyncio.run(scenario())


def test_marker_rectification_module_preserves_video_frame_metadata() -> None:
    async def scenario() -> None:
        frame = VideoFrame(
            image=make_synthetic_marker_image(),
            frame_index=12,
            timestamp_seconds=0.48,
            loop_count=2,
        )
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )

        routed = await module.process(
            Message(frame, metadata={"source": "test"}), AsyncProcessor()
        )

        assert routed is not None
        assert routed.message.payload.shape == (512, 512, 3)
        assert routed.message.metadata["source"] == "test"
        assert routed.message.metadata["frame_index"] == 12
        assert routed.message.metadata["timestamp_seconds"] == 0.48
        assert routed.message.metadata["loop_count"] == 2

    asyncio.run(scenario())


def test_marker_rectification_module_drops_frame_without_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        module = MarkerRectificationModule(
            name="rectifier",
            input_queue="frames",
            output_queue="cutouts",
        )
        blank = np.full((240, 320, 3), 127, dtype=np.uint8)

        with caplog.at_level(logging.WARNING, logger="src.modules.marker_rectifier"):
            routed = await module.process(Message(blank), AsyncProcessor())

        assert routed is None
        assert "Dropping frame without detected marker" in caplog.text

    asyncio.run(scenario())


def test_marker_rectification_module_routes_cutout_to_configured_queue() -> None:
    async def scenario() -> None:
        processor = AsyncProcessor()
        processor.create_queue("frames")
        processor.create_queue("cutouts")
        processor.register_module(
            MarkerRectificationModule(
                name="rectifier",
                input_queue="frames",
                output_queue="cutouts",
            )
        )

        await processor.start()
        await processor.submit("frames", make_synthetic_marker_image())

        result = await asyncio.wait_for(processor.queue("cutouts").get(), timeout=2)
        assert result.payload.shape == (512, 512, 3)
        assert result.payload.dtype == np.uint8
        assert "quad" in result.metadata
        assert "cutout_to_source_homography" in result.metadata
        processor.queue("cutouts").task_done()
        await processor.stop()

    asyncio.run(scenario())


def test_frame_rate_logger_module_uses_loop_count_metadata_for_cutouts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        module = FrameRateLoggerModule(
            name="fps",
            input_queue="marker_cutouts",
            log_interval_seconds=0,
        )
        cutout = np.zeros((32, 32, 3), dtype=np.uint8)

        with caplog.at_level(logging.INFO, logger="src.modules.frame_rate_logger"):
            await module.process(
                Message(cutout, metadata={"loop_count": 7}), AsyncProcessor()
            )

        assert "Processing frame rate" in caplog.text
        assert "source loop 7" in caplog.text

    asyncio.run(scenario())


class FakeFfmpegProcess:
    def __init__(self) -> None:
        self.stdin = self
        self.stderr = None
        self.data = bytearray()
        self.closed = False
        self.waited = False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def close(self) -> None:
        self.closed = True

    def wait(self) -> int:
        self.waited = True
        return 0


class FakeFfmpegStream:
    def __init__(self, capture: dict[str, object], process: FakeFfmpegProcess) -> None:
        self.capture = capture
        self.process = process

    def output(self, path: str, **kwargs: object) -> "FakeFfmpegStream":
        self.capture["output_path"] = path
        self.capture["output_kwargs"] = kwargs
        return self

    def overwrite_output(self) -> "FakeFfmpegStream":
        self.capture["overwrite"] = True
        return self

    def run_async(self, **kwargs: object) -> FakeFfmpegProcess:
        self.capture["run_async_kwargs"] = kwargs
        return self.process


class FakeFfmpeg:
    def __init__(self) -> None:
        self.capture: dict[str, object] = {}
        self.process = FakeFfmpegProcess()

    def input(self, path: str, **kwargs: object) -> FakeFfmpegStream:
        self.capture["input_path"] = path
        self.capture["input_kwargs"] = kwargs
        return FakeFfmpegStream(self.capture, self.process)


def test_ffmpeg_video_writer_streams_rgb_frames_with_expected_encoding_options(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        fake_ffmpeg = FakeFfmpeg()
        module = FfmpegVideoWriterModule(
            name="writer",
            input_queue="frames",
            output_path=tmp_path / "out.mp4",
            fps=24.0,
            ffmpeg_module=fake_ffmpeg,
        )
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[:, :] = np.array([1, 2, 3], dtype=np.uint8)

        await module.process(Message(image), AsyncProcessor())
        await module.close()

        assert fake_ffmpeg.capture["input_path"] == "pipe:"
        assert fake_ffmpeg.capture["input_kwargs"] == {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
            "s": "3x2",
            "framerate": 24.0,
        }
        assert fake_ffmpeg.capture["output_path"] == str(tmp_path / "out.mp4")
        assert fake_ffmpeg.capture["output_kwargs"] == {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "r": 24.0,
        }
        assert fake_ffmpeg.capture["run_async_kwargs"] == {
            "pipe_stdin": True,
            "pipe_stderr": True,
        }
        assert bytes(fake_ffmpeg.process.data[:3]) == bytes([3, 2, 1])
        assert len(fake_ffmpeg.process.data) == 2 * 3 * 3
        assert fake_ffmpeg.process.closed is True
        assert fake_ffmpeg.process.waited is True
        assert module.frames_written == 1

    asyncio.run(scenario())


def test_export_tracker_and_annotator_preserve_frame_without_marker() -> None:
    async def scenario() -> None:
        raw = np.full((80, 100, 3), 77, dtype=np.uint8)
        tracker = OpticalFlowMarkerTrackingModule(
            name="tracker",
            input_queue="frames",
            output_queue="detections",
            emit_empty_detections=True,
        )
        annotator = ArucoMarkerAnnotationModule(
            name="annotator",
            input_queue="detections",
            output_queue="annotated",
        )

        tracked = await tracker.process(Message(raw), AsyncProcessor())
        assert tracked is not None
        assert tracked.message.metadata["marker_ids"] == []
        annotated = await annotator.process(tracked.message, AsyncProcessor())

        assert annotated is not None
        assert annotated.destination == "annotated"
        assert annotated.message.metadata["annotated_marker_count"] == 0
        assert annotated.message.metadata["annotated_marker_ids"] == []
        assert np.array_equal(annotated.message.payload, raw)

    asyncio.run(scenario())


def test_output_video_mode_uses_source_video_fps_and_skips_interrupted_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class SpyFiniteVideoSource:
        source_fps = 12.5

        def __init__(self, path: Path, *, realtime: bool) -> None:
            captured["source_path"] = path
            captured["source_realtime"] = realtime

    class SpyWriter(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_path: Path,
            *,
            fps: float,
        ) -> None:
            super().__init__(name, input_queue)
            captured["writer_input_queue"] = input_queue
            captured["writer_output_path"] = output_path
            captured["writer_fps"] = fps

        async def process(
            self, message: Message[np.ndarray], context: ModuleContext
        ) -> None:
            return None

    class ForbiddenProcessorLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("output video mode must not use ProcessorLoop")

    async def fake_run_finite_export(
        processor: AsyncProcessor,
        source: app_main.InputSource,
        *,
        input_paths: list[Path],
    ) -> None:
        captured["export_source"] = source
        captured["export_input_paths"] = input_paths

    monkeypatch.setattr(app_main, "FiniteVideoSource", SpyFiniteVideoSource)
    monkeypatch.setattr(app_main, "FfmpegVideoWriterModule", SpyWriter)
    monkeypatch.setattr(app_main, "ProcessorLoop", ForbiddenProcessorLoop)
    monkeypatch.setattr(app_main, "run_finite_export", fake_run_finite_export)
    args = app_main.parse_args(
        [
            "--input-path",
            str(TEST_VIDEO_PATH),
            "--output-video",
            str(tmp_path / "out.mp4"),
            "--output-fps",
            "30",
        ]
    )

    asyncio.run(app_main.run_app(args))

    assert captured["source_path"] == TEST_VIDEO_PATH
    assert captured["source_realtime"] is False
    assert captured["writer_input_queue"] == app_main.ANNOTATED_FRAMES_QUEUE
    assert captured["writer_output_path"] == tmp_path / "out.mp4"
    assert captured["writer_fps"] == 12.5
    assert captured["export_input_paths"] == [TEST_VIDEO_PATH]


def test_output_video_mode_uses_default_fps_for_image_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class SpyFiniteImageSource:
        def __init__(self, paths: list[Path]) -> None:
            captured["image_paths"] = paths

    class SpyWriter(BaseModule[np.ndarray]):
        def __init__(
            self,
            name: str,
            input_queue: str,
            output_path: Path,
            *,
            fps: float,
        ) -> None:
            super().__init__(name, input_queue)
            captured["writer_fps"] = fps

        async def process(
            self, message: Message[np.ndarray], context: ModuleContext
        ) -> None:
            return None

    async def fake_run_finite_export(
        processor: AsyncProcessor,
        source: app_main.InputSource,
        *,
        input_paths: list[Path],
    ) -> None:
        captured["export_source"] = source

    monkeypatch.setattr(app_main, "FiniteImageSource", SpyFiniteImageSource)
    monkeypatch.setattr(app_main, "FfmpegVideoWriterModule", SpyWriter)
    monkeypatch.setattr(app_main, "run_finite_export", fake_run_finite_export)
    args = app_main.parse_args(
        [
            "--input-path",
            str(tmp_path / "a.png"),
            str(tmp_path / "b.png"),
            "--output-video",
        ]
    )

    asyncio.run(app_main.run_app(args))

    assert captured["image_paths"] == [tmp_path / "a.png", tmp_path / "b.png"]
    assert captured["writer_fps"] == app_main.DEFAULT_IMAGE_OUTPUT_FPS
