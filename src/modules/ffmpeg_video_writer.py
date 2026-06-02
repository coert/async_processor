from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np

from ..messages import Message
from .base import BaseModule, ModuleContext
from .image_enhancer import validate_color_image

logger = logging.getLogger(__name__)


class FfmpegVideoWriterError(RuntimeError):
    pass


class FfmpegVideoWriterModule(BaseModule[np.ndarray]):
    run_in_thread = True

    def __init__(
        self,
        name: str,
        input_queue: str,
        output_path: str | Path,
        *,
        fps: float,
        ffmpeg_module: Any = ffmpeg,
    ) -> None:
        if not str(output_path):
            raise ValueError("output_path cannot be empty.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        super().__init__(name=name, input_queue=input_queue)
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.ffmpeg = ffmpeg_module
        self.frames_written = 0
        self._process: Any | None = None
        self._frame_shape: tuple[int, int, int] | None = None
        self._closed = False

    async def process(
        self,
        message: Message[np.ndarray],
        context: ModuleContext,
    ) -> None:
        self.process_blocking(message, context)
        return None

    def process_blocking(
        self,
        message: Message[np.ndarray],
        context: ModuleContext,
    ) -> None:
        image = message.payload
        validate_color_image(image)
        if self._frame_shape is None:
            self._frame_shape = tuple(int(value) for value in image.shape)
            self._start_process(width=image.shape[1], height=image.shape[0])
        elif tuple(int(value) for value in image.shape) != self._frame_shape:
            raise FfmpegVideoWriterError(
                f"Video frame shape changed from {self._frame_shape} to {image.shape}."
            )

        if self._process is None or self._process.stdin is None:
            raise FfmpegVideoWriterError("FFmpeg process is not ready for input.")

        rgb_frame = np.ascontiguousarray(image[:, :, ::-1])
        try:
            self._process.stdin.write(rgb_frame.tobytes())
        except BrokenPipeError as exc:
            raise FfmpegVideoWriterError(
                "FFmpeg stopped while writing video frames."
            ) from exc
        self.frames_written += 1
        return None

    def _start_process(self, *, width: int, height: int) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Writing annotated video to %s (%sx%s at %.2f FPS)",
            self.output_path,
            width,
            height,
            self.fps,
        )
        stream = self.ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=self.fps,
        )
        stream = stream.output(
            str(self.output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            r=self.fps,
        )
        self._process = stream.overwrite_output().run_async(
            pipe_stdin=True,
            pipe_stderr=True,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return

        await asyncio.to_thread(self._wait_for_process_exit)

    def _wait_for_process_exit(self) -> None:
        if self._process is None:
            return

        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise FfmpegVideoWriterError(
                f"FFmpeg exited with status {return_code}: {detail}"
            )
        logger.info(
            "Finished writing %s frame(s) to %s",
            self.frames_written,
            self.output_path,
        )
