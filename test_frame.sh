#!/usr/bin/env bash
set -euo pipefail

MARKER_TEMPLATE_DIR="data/aruco/6x6_1000"
INPUT_PATH=${1:-"data/test_frame.jpg"}

if [[ ! -d "${MARKER_TEMPLATE_DIR}" ]]; then
  echo "${MARKER_TEMPLATE_DIR} not found; generating ArUco marker templates..."
  uv run python generate_aruco_markers.py
fi

uv run python main.py --input-path "${INPUT_PATH}" --output-video --debug
