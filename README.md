# async-processor

## Test Frame Helper

For a quick single-frame debug/export run, use the included `test_frame.sh` helper script:

```sh
./test_frame.sh
./test_frame.sh path/to/frame.jpg
```

The annotated-frame overlay uses generated marker template images. Run `uv run python generate_aruco_markers.py` before using this helper, or let `test_frame.sh` do it automatically when `data/aruco/6x6_1000` is missing. It then runs `main.py` through `uv` with `--output-video --debug`, uses `data/test_frame.jpg` by default, and writes the annotated MP4 plus debug images under `data/debug/`.

## Overview

A lightweight `asyncio.Queue` based processing loop. Each module consumes one dedicated input queue and returns routed messages for arbitrary output queues.

The top-level program reads frames from `data/1-input.mp4` by default, or from image files supplied with `--input-path`, enhances them, tracks ArUco marker corners with optical flow, falls back to marker rectification plus ArUco detection when confidence is low, annotates the source frame, loops the input when it ends, and logs processing FPS once per second. If `data/color_classifier_gmm.joblib` exists, the live loop also runs GMM color masking on enhanced frames.

## Run

All commands should go through `uv`:

```sh
uv run python main.py
uv run python -m src
uv run pytest
```

`uv run python main.py` reads `data/1-input.mp4` until `Ctrl+C` or `SIGTERM`, then stops the processor cleanly.

Useful options:

```sh
uv run python main.py --input-path data/1-input.mp4
uv run python main.py --input-path "data/frames/*.png"
uv run python main.py --input-path "data/aruco/video-2/frame_[0006-0010].jpg"
uv run python main.py --input-path data/a.png data/b.jpg "captures/*.jpeg"
uv run python main.py --no-realtime --log-level DEBUG
uv run python main.py --output-video --debug
uv run python main.py --output-video data/debug/custom_annotated.mp4 --output-fps 12
uv run python main.py --debug --no-color
```

Logging is colorized by severity. Use `--log-level DEBUG` for queue/module lifecycle logs, or `--no-color` when writing logs to a plain file. `--output-video` processes the input once with finite image/video sources, writes an annotated MP4, and exits. Without `--output-video`, video and image inputs loop until interrupted.

## Main Pipeline

Without a GMM model:

```text
frames -> image-enhancer -> enhanced_frames -> enhanced-frame-fanout
                                              -> marker_frames -> optical-flow-marker-tracker -> aruco_detections -> aruco-marker-annotator -> annotated_frames -> frame-rate-logger
```

With `data/color_classifier_gmm.joblib` present:

```text
frames -> image-enhancer -> enhanced_frames -> enhanced-frame-fanout
                                              -> marker_frames -> optical-flow-marker-tracker -> aruco_detections -> aruco-marker-annotator -> annotated_frames -> frame-rate-logger
                                              -> gmm_frames    -> gmm-color-mask   -> color_masks
```

For `--output-video`, the frame-rate logger is replaced by `ffmpeg-video-writer`:

```text
frames -> image-enhancer -> enhanced_frames -> enhanced-frame-fanout
                                              -> marker_frames -> optical-flow-marker-tracker -> aruco_detections -> aruco-marker-annotator -> annotated_frames -> ffmpeg-video-writer
```

`color_masks` is intentionally unbounded in the live loop so generated side output does not block the marker pipeline while no downstream consumer is registered yet. The GMM side branch is disabled during video export. The optical-flow marker tracker runs on the marker branch and performs rectification plus ArUco detection internally only when it needs to refresh tracking state.

## Available Modules

### `ImageEnhancementModule`

Applies the underwater image enhancement pipeline to OpenCV-style BGR `uint8` color images. If the payload is a `VideoFrame`, frame metadata is preserved and only `.image` is replaced.

```python
from src import ImageEnhancementModule

processor.create_queue('frames')
processor.create_queue('enhanced_frames')
processor.register_module(
    ImageEnhancementModule(
        name='image-enhancer',
        input_queue='frames',
        output_queue='enhanced_frames',
    )
)
```

### `MarkerRectificationModule`

Detects the marker square in a BGR image and outputs a perspective-rectified square cutout. Frames without a valid marker are dropped with a warning.

```python
from src import MarkerRectificationModule

processor.create_queue('enhanced_frames')
processor.create_queue('marker_cutouts')
processor.register_module(
    MarkerRectificationModule(
        name='marker-rectifier',
        input_queue='enhanced_frames',
        output_queue='marker_cutouts',
        debug=False,
    )
)
```

With `debug=True`, the rectifier overwrites the latest input/line views and writes per-frame quad and cutout images under `data/debug/`:

```text
marker_input.png
marker_hough_lines.png
marker_detected_quad_####.png
marker_rectified_cutout_####.png
```

### `GMMColorMaskModule`

Loads `data/color_classifier_gmm.joblib` and converts BGR images into single-channel black/white masks. The model compares the likelihood of each pixel under two Gaussian mixture models: one trained on the target pipeline color and one trained on background/non-pipeline samples. For a concise visual explanation of color classification with Gaussian mixture models, see the CMSC426 color segmentation notes: <https://cmsc426.github.io/colorseg/>.

The model must be a joblib dict with `query_gmm`, `non_query_gmm`, `query_prior`, and `non_query_prior`.

```python
from pathlib import Path
from src import GMMColorMaskModule

processor.create_queue('gmm_frames')
processor.create_queue('color_masks')
processor.register_module(
    GMMColorMaskModule(
        name='gmm-color-mask',
        input_queue='gmm_frames',
        output_queue='color_masks',
        model_path=Path('data/color_classifier_gmm.joblib'),
        debug=False,
    )
)
```

With `debug=True`, the latest mask is overwritten at:

```text
data/debug/gmm_color_mask.png
```

### `OpticalFlowMarkerTrackingModule`

Tracks the detected outer/source quad frame-to-frame with pyramidal Lucas-Kanade optical flow, then runs `MarkerRectificationModule` and `ArucoDetectionModule` on every frame using that tracked quad as a prior. When quad tracking fails or confidence drops too low, it forces the rectifier to re-run the full edge and Hough search on that same frame instead of trusting the propagated quad. The output remains detector-backed ArUco detections with the rectifier homography metadata needed by `ArucoMarkerAnnotationModule`.

```python
from src import OpticalFlowMarkerTrackingModule

processor.create_queue('marker_frames')
processor.create_queue('aruco_detections')
processor.register_module(
    OpticalFlowMarkerTrackingModule(
        name='optical-flow-marker-tracker',
        input_queue='marker_frames',
        output_queue='aruco_detections',
        max_forward_error=25.0,
        max_backtrack_error=3.0,
        min_marker_area=16.0,
        debug=False,
    )
)
```

### `ArucoDetectionModule`

Detects OpenCV ArUco markers in rectified marker cutouts using a predefined dictionary. The detector runs once on the raw cutout and once with a white border so raw-only hits are preserved while tightly cropped cutouts still get a quiet margin. When markers are found, it outputs an `ArucoDetectionResult` containing the input image, marker ids, marker corners, and rejected candidates. Frames without markers are dropped.

```python
from src import ArucoDetectionModule

processor.create_queue('marker_cutouts')
processor.create_queue('aruco_detections')
processor.register_module(
    ArucoDetectionModule(
        name='aruco-detector',
        input_queue='marker_cutouts',
        output_queue='aruco_detections',
        dictionary_name='DICT_6X6_1000',
        input_border_pixels=16,
        debug=False,
    )
)
```

With `debug=True`, the detector overwrites the latest raw cutout input and writes per-frame overlay images under `data/debug/`:

```text
aruco_input.png
aruco_detected_markers_####.png  # union of raw and padded detections on the padded input canvas
aruco_rejected_candidates_####.png
```

### `ArucoMarkerAnnotationModule`

Draws detected marker ids onto the original source frame and overlays a compact marker-template grid in the lower-right corner when template images are available under `data/aruco/6x6_1000`. It expects detection metadata containing the source frame image and the cutout-to-source homography, which is provided by the rectifier/tracker path used by `main.py`.

```python
from src import ArucoMarkerAnnotationModule

processor.create_queue('aruco_detections')
processor.create_queue('annotated_frames')
processor.register_module(
    ArucoMarkerAnnotationModule(
        name='aruco-marker-annotator',
        input_queue='aruco_detections',
        output_queue='annotated_frames',
        debug=False,
    )
)
```

With `debug=True`, the annotator overwrites:

```text
data/debug/aruco_annotated_frame.png
```

### `FfmpegVideoWriterModule`

Consumes BGR image frames and writes them to an MP4 using FFmpeg. The first frame fixes the output resolution; later frames must have the same shape. `main.py --output-video` wires this module to `annotated_frames` and closes it after the finite source is exhausted.

```python
from pathlib import Path
from src import FfmpegVideoWriterModule

processor.create_queue('annotated_frames')
processor.register_module(
    FfmpegVideoWriterModule(
        name='ffmpeg-video-writer',
        input_queue='annotated_frames',
        output_path=Path('data/debug/aruco_annotated_video.mp4'),
        fps=30.0,
    )
)
```

### `QueueFanoutModule`

Routes one input message to multiple output queues. This is useful because each queue can have only one consuming module.

```python
from src import QueueFanoutModule

processor.create_queue('enhanced_frames')
processor.create_queue('marker_frames')
processor.create_queue('gmm_frames')
processor.register_module(
    QueueFanoutModule(
        name='enhanced-frame-fanout',
        input_queue='enhanced_frames',
        output_queues=['marker_frames', 'gmm_frames'],
    )
)
```

### `FrameRateLoggerModule`

Consumes any payload and logs processing rate once per interval. It reads `loop_count` from `VideoFrame` payloads or message metadata when available.

```python
from src import FrameRateLoggerModule

processor.create_queue('marker_cutouts')
processor.register_module(
    FrameRateLoggerModule(
        name='frame-rate-logger',
        input_queue='marker_cutouts',
    )
)
```

### `LoopingVideoSource`

Input source that reads frames from a video file and loops back to the first frame at EOF. It produces `VideoFrame` payloads. Use `FiniteVideoSource` when processing a video once for export.

```python
from src import LoopingVideoSource, ProcessorLoop

source = LoopingVideoSource('data/1-input.mp4', realtime=True)
runner = ProcessorLoop(processor, input_queue='frames', source=source)
await runner.run_until_interrupted()
```

### `LoopingImageSource`

Input source that reads one or more image files, expands glob patterns or numeric ranges such as `frame_[0006-0010].jpg`, and loops back to the first image after the set ends. It produces `ImageFrame` payloads. Use `FiniteImageSource` when processing an image set once for export.

```python
from src import LoopingImageSource, ProcessorLoop

source = LoopingImageSource(["data/a.png", "data/b.jpg", "captures/*.jpeg", "frames/frame_[0006-0010].jpg"])
runner = ProcessorLoop(processor, input_queue="frames", source=source)
await runner.run_until_interrupted()
```

## Color Sampler

`color_sampler.py` is a separate program for creating the GMM model used by `GMMColorMaskModule`. You only mark the `pipeline` foreground. Every unmarked pixel in the loaded image is treated as background during training.

Run it with:

```sh
uv run python color_sampler.py --image data/debug/marker_input.png
```

On a remote/headless machine, it automatically starts a browser sampler instead of OpenCV/Qt:

```text
Browser color sampler running at http://127.0.0.1:8765/
```

Open that URL through SSH or VS Code port forwarding. You can force browser mode or choose a port explicitly:

```sh
uv run python color_sampler.py --image data/debug/marker_input.png --web --port 8765
```

Controls:

```text
left drag = mark pipeline foreground pixels
+ / -     = brush size
s         = save CSV samples
t         = train and save data/color_classifier_gmm.joblib
q         = quit OpenCV GUI mode
```

Default outputs:

```text
data/color_samples.csv
data/color_classifier_gmm.joblib
```

The CSV stores label, pixel coordinates, RGB, BGR, and Lab values. The joblib model is inference-compatible with the main loop; once it exists, `main.py` automatically enables the GMM mask module.


## ArUco Marker Generator

`generate_aruco_markers.py` converts the vendored arucogen `data/aruco_dict.json` into PNG marker images. By default it writes every standard non-AprilTag dictionary to `data/aruco/` and creates `data/aruco/manifest.csv`.

```sh
uv run python generate_aruco_markers.py
```

Useful options:

```sh
uv run python generate_aruco_markers.py --dictionary 4x4_1000
uv run python generate_aruco_markers.py --image-size 1024
uv run python generate_aruco_markers.py --include-apriltag
```

The default output includes `aruco`, `4x4_1000`, `5x5_1000`, `6x6_1000`, `7x7_1000`, and `mip_36h12`. Marker images are grouped by dictionary name under `data/aruco/`.

## Minimal Module

```python
from src import BaseModule, Message, ModuleContext, RoutedMessage


class UppercaseModule(BaseModule[str]):
    async def process(
        self,
        message: Message[str],
        context: ModuleContext,
    ) -> RoutedMessage[str]:
        return RoutedMessage.from_payload('results', message.payload.upper())
```

## Minimal Processor Loop

```python
from src import AsyncProcessor, ProcessorLoop

processor = AsyncProcessor()
processor.create_queue('input')
processor.create_queue('results')
processor.register_module(UppercaseModule(name='uppercase', input_queue='input'))

runner = ProcessorLoop(processor, input_queue='input', source=my_frame_source)
await runner.run_until_interrupted()
```

A source only needs an async `poll()` method. Return `None` when no input is currently available, a `Message`, or a raw payload that should be wrapped in `Message` automatically.
