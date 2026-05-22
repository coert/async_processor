#!/usr/bin/env bash
set -e

INPUT_PATH=${1:-"data/test_frame.jpg"}

uv run python main.py --input-path ${INPUT_PATH} --output-video --debug