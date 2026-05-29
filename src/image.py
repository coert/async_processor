from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

logger = logging.getLogger(__name__)

NUMERIC_RANGE_PATTERN = re.compile(r"\[(\d+)-(\d+)\]")

IMAGE_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)


class ImageSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ImageFrame:
    image: Any
    frame_index: int
    timestamp_seconds: float
    loop_count: int
    path: Path


class LoopingImageSource:
    def __init__(self, paths: str | Path | Iterable[str | Path]) -> None:
        self.paths = self._resolve_paths(paths)
        if not self.paths:
            raise ImageSourceError("No image inputs matched.")
        self.frame_indices = self._frame_indices_for_paths(self.paths)

        self.frame_count = len(self.paths)
        self.loop_count = 0
        self._index = 0

        logger.info("Opened %s image input(s)", self.frame_count)

    async def poll(self) -> ImageFrame:
        path = self.paths[self._index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageSourceError(f"Could not read image file: {path}")

        frame = ImageFrame(
            image=image,
            frame_index=self.frame_indices[self._index],
            timestamp_seconds=0.0,
            loop_count=self.loop_count,
            path=path,
        )

        self._index += 1
        if self._index >= self.frame_count:
            self._index = 0
            self.loop_count += 1
            logger.info("Image set ended; looping from the first image")

        return frame

    def close(self) -> None:
        return None

    @classmethod
    def _resolve_paths(
        cls, paths: str | Path | Iterable[str | Path]
    ) -> tuple[Path, ...]:
        if isinstance(paths, (str, Path)):
            raw_paths = [paths]
        else:
            raw_paths = list(paths)

        resolved: list[Path] = []
        unmatched_patterns: list[str] = []
        for raw_path in raw_paths:
            path_text = str(raw_path)
            if cls._contains_numeric_range_syntax(path_text):
                resolved.extend(cls._expand_numeric_range(path_text))
            elif glob.has_magic(path_text):
                matches = sorted(Path(match) for match in glob.glob(path_text))
                if not matches:
                    unmatched_patterns.append(path_text)
                resolved.extend(matches)
            else:
                resolved.append(Path(raw_path))

        if unmatched_patterns:
            patterns = ", ".join(unmatched_patterns)
            raise ImageSourceError(
                f"No image inputs matched glob pattern(s): {patterns}"
            )

        image_paths = tuple(
            path for path in resolved if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(image_paths) != len(resolved):
            unsupported = sorted(
                str(path)
                for path in resolved
                if path.suffix.lower() not in IMAGE_EXTENSIONS
            )
            raise ImageSourceError(
                "Unsupported image input extension(s): " + ", ".join(unsupported)
            )

        return image_paths

    @staticmethod
    def _frame_indices_for_paths(paths: tuple[Path, ...]) -> tuple[int, ...]:
        return tuple(
            LoopingImageSource._frame_index_from_path(path, fallback=index)
            for index, path in enumerate(paths)
        )

    @staticmethod
    def _frame_index_from_path(path: Path, fallback: int) -> int:
        matches = re.findall(r"(\d+)", path.stem)
        if not matches:
            return fallback
        return int(matches[-1])

    @classmethod
    def _contains_numeric_range_syntax(cls, path_text: str) -> bool:
        if NUMERIC_RANGE_PATTERN.search(path_text):
            return True
        if path_text.count("[") != path_text.count("]"):
            return True
        bracketed_parts = re.findall(r"\[[^\]]*\]", path_text)
        return any("-" in part for part in bracketed_parts)

    @classmethod
    def _expand_numeric_range(cls, path_text: str) -> tuple[Path, ...]:
        matches = list(NUMERIC_RANGE_PATTERN.finditer(path_text))
        if len(matches) != 1:
            raise ImageSourceError(
                f"Input path must contain exactly one numeric range: {path_text}"
            )

        match = matches[0]
        start_text, end_text = match.groups()
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ImageSourceError(f"Numeric range cannot descend: {path_text}")

        prefix = path_text[: match.start()]
        suffix = path_text[match.end() :]
        width = (
            max(len(start_text), len(end_text))
            if start_text.startswith("0") or end_text.startswith("0")
            else 0
        )
        expanded_paths = [
            Path(
                f"{prefix}{value:0{width}d}{suffix}"
                if width
                else f"{prefix}{value}{suffix}"
            )
            for value in range(start, end + 1)
        ]
        unsupported_paths = [
            path
            for path in expanded_paths
            if path.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        if unsupported_paths:
            unsupported = ", ".join(str(path) for path in unsupported_paths)
            raise ImageSourceError(
                f"Unsupported image input extension(s): {unsupported}"
            )

        image_paths = [
            path
            for path in expanded_paths
            if path.exists() and cv2.imread(str(path), cv2.IMREAD_COLOR) is not None
        ]
        if not image_paths:
            raise ImageSourceError(
                f"No readable image inputs matched numeric range: {path_text}"
            )
        if len(image_paths) != len(expanded_paths):
            logger.info(
                "Skipped %s missing or unreadable image(s) from numeric range %s",
                len(expanded_paths) - len(image_paths),
                path_text,
            )

        return tuple(image_paths)


class FiniteImageSource(LoopingImageSource):
    async def poll(self) -> ImageFrame | None:
        if self._index >= self.frame_count:
            logger.info("Image set ended; finite source exhausted")
            return None

        path = self.paths[self._index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageSourceError(f"Could not read image file: {path}")

        frame = ImageFrame(
            image=image,
            frame_index=self.frame_indices[self._index],
            timestamp_seconds=0.0,
            loop_count=0,
            path=path,
        )
        self._index += 1
        return frame
