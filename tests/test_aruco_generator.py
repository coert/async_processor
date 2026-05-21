from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from generate_aruco_markers import (
    generate_markers,
    load_marker_dictionaries,
    marker_size_for_dictionary,
    render_marker_image,
    unpack_marker_bits,
)


def test_marker_size_is_inferred_for_standard_dictionaries() -> None:
    assert marker_size_for_dictionary("aruco") == 5
    assert marker_size_for_dictionary("4x4_1000") == 4
    assert marker_size_for_dictionary("7x7_1000") == 7
    assert marker_size_for_dictionary("mip_36h12") == 6


@pytest.mark.parametrize("name", ["unknown", "4x5_1000"])
def test_marker_size_rejects_unknown_dictionary_names(name: str) -> None:
    with pytest.raises(ValueError, match="Cannot infer"):
        marker_size_for_dictionary(name)


def test_unpack_marker_bits_matches_arucogen_partial_byte_order() -> None:
    bits = unpack_marker_bits([0b1010_0101, 0b0000_0001], marker_size=3)

    assert bits.tolist() == [
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
    ]


def test_render_marker_image_adds_black_border_and_white_cells() -> None:
    image = render_marker_image(
        [0b1010_0101, 0b1100_0011],
        marker_size=4,
        image_size=12,
        border_bits=1,
    )

    assert image.shape == (12, 12)
    assert image.dtype == np.uint8
    assert image[0, 0] == 0
    assert image[2, 2] == 255
    assert image[2, 4] == 0
    assert image[4, 2] == 0
    assert set(np.unique(image)) == {0, 255}


def test_render_marker_image_supports_arbitrary_output_size() -> None:
    image = render_marker_image([0, 0], marker_size=4, image_size=17, border_bits=1)

    assert image.shape == (17, 17)


def test_load_marker_dictionaries_defaults_to_standard_non_apriltag(tmp_path) -> None:
    dict_json = tmp_path / "dict.json"
    dict_json.write_text(
        json.dumps(
            {
                "aruco": [[0, 0, 0, 0]],
                "4x4_1000": [[0, 0]],
                "april_16h5": [[0, 0]],
            }
        )
    )

    dictionaries = load_marker_dictionaries(dict_json)

    assert [dictionary.name for dictionary in dictionaries] == ["aruco", "4x4_1000"]


def test_generate_markers_writes_pngs_and_manifest(tmp_path) -> None:
    dict_json = tmp_path / "dict.json"
    dict_json.write_text(json.dumps({"4x4_1000": [[0, 0], [255, 255]]}))
    output_dir = tmp_path / "aruco"
    dictionaries = load_marker_dictionaries(dict_json, selected=["4x4_1000"])

    generated = generate_markers(dictionaries, output_dir, image_size=12, border_bits=1)

    assert len(generated) == 2
    first_image = output_dir / "4x4_1000" / "4x4_1000_0000.png"
    second_image = output_dir / "4x4_1000" / "4x4_1000_0001.png"
    assert first_image.exists()
    assert second_image.exists()
    assert cv2.imread(str(first_image), cv2.IMREAD_GRAYSCALE).shape == (12, 12)

    with (output_dir / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["dictionary"] for row in rows] == ["4x4_1000", "4x4_1000"]
    assert [row["marker_id"] for row in rows] == ["0", "1"]
