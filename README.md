# async-processor

A lightweight `asyncio.Queue` based processing loop. Each module consumes one dedicated input queue and returns routed messages for arbitrary output queues.

The top-level program reads frames from `data/1-input.mp4` by default, enhances them, rectifies detected marker cutouts, loops the video when it ends, and logs processing FPS once per second. If `data/color_classifier_gmm.joblib` exists, the loop also runs GMM color masking on enhanced frames.

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
uv run python main.py --video-path data/1-input.mp4
uv run python main.py --no-realtime --log-level DEBUG
uv run python main.py --debug --no-color
```

Logging is colorized by severity. Use `--log-level DEBUG` for queue/module lifecycle logs, or `--no-color` when writing logs to a plain file.

## Main Pipeline

Without a GMM model:

```text
frames -> image-enhancer -> enhanced_frames -> marker-rectifier -> marker_cutouts -> frame-rate-logger
```

With `data/color_classifier_gmm.joblib` present:

```text
frames -> image-enhancer -> enhanced_frames -> enhanced-frame-fanout
                                              -> marker_frames -> marker-rectifier -> marker_cutouts -> frame-rate-logger
                                              -> gmm_frames    -> gmm-color-mask   -> color_masks
```

`color_masks` is intentionally unbounded in the main loop so mask generation does not block the marker pipeline while no downstream consumer is registered yet.

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

With `debug=True`, the rectifier overwrites these files under `data/debug/`:

```text
marker_input.png
marker_hough_lines.png
marker_detected_quad.png
marker_rectified_cutout.png
```

### `GMMColorMaskModule`

Loads `data/color_classifier_gmm.joblib` and converts BGR images into single-channel black/white masks. The model must be a joblib dict with `query_gmm`, `non_query_gmm`, `query_prior`, and `non_query_prior`.

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

Input source that reads frames from a video file and loops back to the first frame at EOF. It produces `VideoFrame` payloads.

```python
from src import LoopingVideoSource, ProcessorLoop

source = LoopingVideoSource('data/1-input.mp4', realtime=True)
runner = ProcessorLoop(processor, input_queue='frames', source=source)
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
