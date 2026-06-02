from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
from pathlib import Path
from typing import Sequence

from src import (
    ArucoMarkerAnnotationModule,
    AsyncProcessor,
    FfmpegVideoWriterModule,
    FiniteImageSource,
    FiniteVideoSource,
    FrameRateLoggerModule,
    GMMColorMaskModule,
    ImageEnhancementModule,
    InputSource,
    LoopingImageSource,
    LoopingVideoSource,
    OpticalFlowMarkerTrackingModule,
    ProcessorLoop,
    QueueFanoutModule,
    configure_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path(__file__).parent / "data" / "1-input.mp4"
DEFAULT_OUTPUT_VIDEO_PATH = Path("data/debug/aruco_annotated_video.mp4")
DEFAULT_IMAGE_OUTPUT_FPS = 30.0
VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".webm",
    }
)
FRAME_QUEUE = "frames"
ENHANCED_FRAME_QUEUE = "enhanced_frames"
GMM_MODEL_PATH = Path("data/color_classifier_gmm.joblib")
GMM_FRAME_QUEUE = "gmm_frames"
MARKER_FRAME_QUEUE = "marker_frames"
COLOR_MASK_QUEUE = "color_masks"
ARUCO_DETECTIONS_QUEUE = "aruco_detections"
ANNOTATED_FRAMES_QUEUE = "annotated_frames"
EXPORT_VIDEO_QUEUE = "export_video_frames"
EXPORT_FPS_LOG_QUEUE = "export_fps_log_frames"


def is_video_input(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def create_input_source(input_paths: Sequence[Path], *, realtime: bool) -> InputSource:
    if len(input_paths) == 1 and is_video_input(input_paths[0]):
        return LoopingVideoSource(input_paths[0], realtime=realtime)
    return LoopingImageSource(input_paths)


def create_finite_input_source(
    input_paths: Sequence[Path],
    *,
    image_output_fps: float,
) -> tuple[InputSource, float]:
    if len(input_paths) == 1 and is_video_input(input_paths[0]):
        source = FiniteVideoSource(input_paths[0], realtime=False)
        output_fps = source.source_fps if source.source_fps > 0 else image_output_fps
        return source, output_fps

    return FiniteImageSource(input_paths), image_output_fps


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the async processor loop.")
    parser.add_argument(
        "--input-path",
        default=[DEFAULT_INPUT_PATH],
        nargs="+",
        type=Path,
        help="Video file, image file(s), or image glob pattern(s) to poll as the input source.",
    )
    parser.add_argument(
        "--queue-size",
        default=2,
        type=int,
        help="Maximum number of queued frames waiting for processing.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Read video frames as fast as processing allows instead of at video FPS.",
    )
    parser.add_argument(
        "--output-video",
        nargs="?",
        const=DEFAULT_OUTPUT_VIDEO_PATH,
        default=None,
        type=Path,
        help="Write a finite annotated MP4 to this path and exit.",
    )
    parser.add_argument(
        "--output-fps",
        default=DEFAULT_IMAGE_OUTPUT_FPS,
        type=float,
        help="Output FPS for image-set video export, or fallback FPS when video FPS is unavailable.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum log level to show.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in log output.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write processing debug images to data/debug/.",
    )
    parser.add_argument(
        "--dictionary-name",
        default="DICT_6x6_1000",
        help="Name of the ArUco marker dictionary to use, e.g. DICT_6x6_1000.",
    )
    return parser.parse_args(argv)


def register_processing_modules(
    processor: AsyncProcessor,
    args: argparse.Namespace,
    *,
    include_gmm: bool,
    emit_empty_detections: bool,
) -> None:
    processor.create_queue(FRAME_QUEUE, maxsize=args.queue_size)
    processor.create_queue(ENHANCED_FRAME_QUEUE, maxsize=args.queue_size)
    processor.create_queue(MARKER_FRAME_QUEUE, maxsize=args.queue_size)
    processor.create_queue(ARUCO_DETECTIONS_QUEUE)
    processor.create_queue(ANNOTATED_FRAMES_QUEUE)
    if include_gmm:
        processor.create_queue(GMM_FRAME_QUEUE, maxsize=args.queue_size)
        processor.create_queue(COLOR_MASK_QUEUE)

    processor.register_module(
        ImageEnhancementModule(
            name="image-enhancer",
            input_queue=FRAME_QUEUE,
            output_queue=ENHANCED_FRAME_QUEUE,
            debug=args.debug,
            debug_dir=Path("data/debug/image-enhancer"),
        )
    )
    fanout_output_queues = [MARKER_FRAME_QUEUE]
    if include_gmm:
        fanout_output_queues.append(GMM_FRAME_QUEUE)
    processor.register_module(
        QueueFanoutModule(
            name="enhanced-frame-fanout",
            input_queue=ENHANCED_FRAME_QUEUE,
            output_queues=fanout_output_queues,
        )
    )

    if include_gmm:
        processor.register_module(
            GMMColorMaskModule(
                name="gmm-color-mask",
                input_queue=GMM_FRAME_QUEUE,
                output_queue=COLOR_MASK_QUEUE,
                model_path=GMM_MODEL_PATH,
                debug=args.debug,
                debug_dir=Path("data/debug/gmm-color-mask"),
            )
        )
        logger.info("GMM color mask module enabled with model %s", GMM_MODEL_PATH)
    else:
        logger.info("GMM color mask module disabled for this run")

    processor.register_module(
        OpticalFlowMarkerTrackingModule(
            name="optical-flow-marker-tracker",
            input_queue=MARKER_FRAME_QUEUE,
            output_queue=ARUCO_DETECTIONS_QUEUE,
            debug=args.debug,
            debug_dir=Path("data/debug/optical-flow-marker-tracker"),
            dictionary_name=args.dictionary_name,
            emit_empty_detections=emit_empty_detections,
        )
    )
    logger.info(
        "Optical-flow marker tracker enabled on input queue %s", MARKER_FRAME_QUEUE
    )
    processor.register_module(
        ArucoMarkerAnnotationModule(
            name="aruco-marker-annotator",
            input_queue=ARUCO_DETECTIONS_QUEUE,
            output_queue=ANNOTATED_FRAMES_QUEUE,
            debug=args.debug,
            debug_dir=Path("data/debug/aruco-marker-annotator"),
            dictionary_name=args.dictionary_name,
        )
    )


async def run_finite_export(
    processor: AsyncProcessor,
    source: InputSource,
    *,
    input_paths: Sequence[Path],
) -> None:
    logger.info(
        "Exporting annotated video from %s",
        ", ".join(str(path) for path in input_paths),
    )
    await processor.start()
    try:
        while True:
            processor.raise_for_failed_tasks()
            item = await source.poll()
            processor.raise_for_failed_tasks()
            if item is None:
                break
            await processor.submit(FRAME_QUEUE, item)

        for queue_name in (
            FRAME_QUEUE,
            ENHANCED_FRAME_QUEUE,
            MARKER_FRAME_QUEUE,
            ARUCO_DETECTIONS_QUEUE,
            ANNOTATED_FRAMES_QUEUE,
        ):
            await processor.queue(queue_name).join()
        processor.raise_for_failed_tasks()
    finally:
        await close_source(source)
        await processor.stop()


async def close_source(source: InputSource) -> None:
    close = getattr(source, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def run_app(args: argparse.Namespace) -> None:
    if args.output_fps <= 0:
        raise ValueError("output_fps must be greater than zero.")

    processor = AsyncProcessor()
    exporting_video = args.output_video is not None
    gmm_model_exists = GMM_MODEL_PATH.exists()
    include_gmm = gmm_model_exists and not exporting_video
    register_processing_modules(
        processor,
        args,
        include_gmm=include_gmm,
        emit_empty_detections=exporting_video,
    )

    if exporting_video:
        source, output_fps = create_finite_input_source(
            args.input_path,
            image_output_fps=args.output_fps,
        )
        processor.create_queue(EXPORT_VIDEO_QUEUE, maxsize=args.queue_size)
        processor.create_queue(EXPORT_FPS_LOG_QUEUE, maxsize=args.queue_size)
        processor.register_module(
            QueueFanoutModule(
                name="annotated-frame-fanout",
                input_queue=ANNOTATED_FRAMES_QUEUE,
                output_queues=[EXPORT_VIDEO_QUEUE, EXPORT_FPS_LOG_QUEUE],
            )
        )
        processor.register_module(
            FrameRateLoggerModule(
                name="frame-rate-logger",
                input_queue=EXPORT_FPS_LOG_QUEUE,
            )
        )
        processor.register_module(
            FfmpegVideoWriterModule(
                name="ffmpeg-video-writer",
                input_queue=EXPORT_VIDEO_QUEUE,
                output_path=args.output_video,
                fps=output_fps,
            )
        )
        await run_finite_export(processor, source, input_paths=args.input_path)
        logger.info("Annotated video export finished: %s", args.output_video)
        return

    if not gmm_model_exists:
        logger.info(
            "GMM color classifier model not found at %s; module disabled",
            GMM_MODEL_PATH,
        )
    processor.register_module(
        FrameRateLoggerModule(
            name="frame-rate-logger",
            input_queue=ANNOTATED_FRAMES_QUEUE,
        )
    )

    source = create_input_source(args.input_path, realtime=not args.no_realtime)
    runner = ProcessorLoop(
        processor,
        input_queue=FRAME_QUEUE,
        source=source,
        poll_interval=0,
    )

    logger.info(
        "Reading frames from %s. Press Ctrl+C to stop.",
        ", ".join(str(path) for path in args.input_path),
    )
    await runner.run_until_interrupted()
    logger.info("Async processor loop stopped cleanly.")


def main() -> None:
    args = parse_args()
    log_level = "DEBUG" if args.debug else args.log_level
    configure_logging(log_level, use_colors=not args.no_color)
    asyncio.run(run_app(args))


if __name__ == "__main__":
    main()
