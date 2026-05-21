from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

DEFAULT_DICT_JSON = Path("data/aruco_dict.json")
DEFAULT_OUTPUT_DIR = Path("data/aruco")
DEFAULT_IMAGE_SIZE = 512
DEFAULT_BORDER_BITS = 1

STANDARD_DICTIONARIES = (
    "aruco",
    "4x4_1000",
    "5x5_1000",
    "6x6_1000",
    "7x7_1000",
    "mip_36h12",
)

EXPLICIT_MARKER_SIZES = {
    "aruco": 5,
    "mip_36h12": 6,
    "april_16h5": 4,
    "april_25h9": 5,
    "april_36h10": 6,
    "april_36h11": 6,
}


@dataclass(frozen=True)
class MarkerDictionary:
    name: str
    marker_size: int
    markers: list[list[int]]


@dataclass(frozen=True)
class GeneratedMarker:
    dictionary: str
    marker_id: int
    path: Path
    image_size: int
    marker_size: int
    border_bits: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG images for every ArUco marker in data/aruco_dict.json."
    )
    parser.add_argument(
        "--dict-json",
        type=Path,
        default=DEFAULT_DICT_JSON,
        help="Path to the arucogen dict.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where marker PNGs and manifest.csv are written.",
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        help=(
            "Dictionary name to generate. Can be passed multiple times. "
            "Defaults to all standard non-AprilTag dictionaries."
        ),
    )
    parser.add_argument(
        "--include-apriltag",
        action="store_true",
        help="Also generate AprilTag dictionaries from dict.json.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output image size in pixels. Must divide evenly by marker cells plus border.",
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=DEFAULT_BORDER_BITS,
        help="Black border width around the marker, in marker cells.",
    )
    return parser.parse_args(argv)


def marker_size_for_dictionary(name: str) -> int:
    if name in EXPLICIT_MARKER_SIZES:
        return EXPLICIT_MARKER_SIZES[name]

    match = re.match(r"^(\d+)x\1_", name)
    if match:
        return int(match.group(1))

    raise ValueError(f"Cannot infer marker size for dictionary {name!r}")


def dictionary_names(data: dict[str, object], include_apriltag: bool = False) -> list[str]:
    names = [name for name in STANDARD_DICTIONARIES if name in data]
    if include_apriltag:
        names.extend(
            name
            for name in data
            if name.startswith("april_") and name not in names
        )
    return names


def load_marker_dictionaries(
    dict_json: Path,
    selected: Iterable[str] | None = None,
    *,
    include_apriltag: bool = False,
) -> list[MarkerDictionary]:
    data = json.loads(dict_json.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{dict_json} must contain a JSON object")

    names = list(selected) if selected else dictionary_names(data, include_apriltag)
    if not names:
        raise ValueError("No marker dictionaries selected")

    dictionaries: list[MarkerDictionary] = []
    for name in names:
        raw_markers = data.get(name)
        if not isinstance(raw_markers, list):
            raise ValueError(f"Dictionary {name!r} is missing or is not a list")

        markers: list[list[int]] = []
        for marker_id, raw_marker in enumerate(raw_markers):
            if not isinstance(raw_marker, list):
                raise ValueError(f"Marker {name}:{marker_id} is not a byte list")
            markers.append([int(byte) for byte in raw_marker])

        dictionaries.append(
            MarkerDictionary(
                name=name,
                marker_size=marker_size_for_dictionary(name),
                markers=markers,
            )
        )
    return dictionaries


def unpack_marker_bits(marker_bytes: Sequence[int], marker_size: int) -> np.ndarray:
    """Decode marker bytes with the same bit order as arucogen/main.js."""
    bit_count = marker_size * marker_size
    bits: list[int] = []

    for byte in marker_bytes:
        remaining = bit_count - len(bits)
        if remaining <= 0:
            break
        for bit_index in range(min(7, remaining - 1), -1, -1):
            bits.append((int(byte) >> bit_index) & 1)

    if len(bits) != bit_count:
        raise ValueError(
            f"Marker has {len(bits)} decoded bits, expected {bit_count} for {marker_size}x{marker_size}"
        )

    return np.array(bits, dtype=np.uint8).reshape(marker_size, marker_size)


def render_marker_image(
    marker_bytes: Sequence[int],
    marker_size: int,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    border_bits: int = DEFAULT_BORDER_BITS,
) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if border_bits < 0:
        raise ValueError("border_bits must be non-negative")

    cells = marker_size + (2 * border_bits)
    cell_grid = np.zeros((cells, cells), dtype=np.uint8)
    marker_bits = unpack_marker_bits(marker_bytes, marker_size)
    cell_grid[
        border_bits : border_bits + marker_size,
        border_bits : border_bits + marker_size,
    ] = marker_bits * 255
    return cv2.resize(
        cell_grid,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )


def marker_filename(dictionary_name: str, marker_id: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dictionary_name).strip("_")
    return f"{safe_name}_{marker_id:04d}.png"


def generate_markers(
    dictionaries: Iterable[MarkerDictionary],
    output_dir: Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    border_bits: int = DEFAULT_BORDER_BITS,
) -> list[GeneratedMarker]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedMarker] = []

    for dictionary in dictionaries:
        dictionary_dir = output_dir / dictionary.name
        dictionary_dir.mkdir(parents=True, exist_ok=True)

        for marker_id, marker_bytes in enumerate(dictionary.markers):
            image = render_marker_image(
                marker_bytes,
                dictionary.marker_size,
                image_size=image_size,
                border_bits=border_bits,
            )
            path = dictionary_dir / marker_filename(dictionary.name, marker_id)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Failed to write marker image {path}")
            generated.append(
                GeneratedMarker(
                    dictionary=dictionary.name,
                    marker_id=marker_id,
                    path=path,
                    image_size=image_size,
                    marker_size=dictionary.marker_size,
                    border_bits=border_bits,
                )
            )

    write_manifest(output_dir / "manifest.csv", generated)
    return generated


def write_manifest(path: Path, generated: Sequence[GeneratedMarker]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dictionary",
                "marker_id",
                "path",
                "image_size",
                "marker_size",
                "border_bits",
            ),
        )
        writer.writeheader()
        for marker in generated:
            writer.writerow(
                {
                    "dictionary": marker.dictionary,
                    "marker_id": marker.marker_id,
                    "path": marker.path.as_posix(),
                    "image_size": marker.image_size,
                    "marker_size": marker.marker_size,
                    "border_bits": marker.border_bits,
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dictionaries = load_marker_dictionaries(
        args.dict_json,
        selected=args.dictionary,
        include_apriltag=args.include_apriltag,
    )
    generated = generate_markers(
        dictionaries,
        args.output_dir,
        image_size=args.image_size,
        border_bits=args.border_bits,
    )
    dictionary_summary = ", ".join(
        f"{dictionary.name}={len(dictionary.markers)}" for dictionary in dictionaries
    )
    print(
        f"Generated {len(generated)} marker PNGs in {args.output_dir} "
        f"({dictionary_summary})"
    )


if __name__ == "__main__":
    main()
