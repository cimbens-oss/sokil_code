#!/bin/bash
# FAST-LIVO2 Python - Run script
# Usage: ./run.sh <path_to_rosbag> [config_yaml] [camera_yaml] [output_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BAG_PATH="${1:?Usage: $0 <rosbag_path> [config_yaml] [camera_yaml] [output_dir]}"
CONFIG="${2:-$SCRIPT_DIR/../FAST-LIVO2-main/config/avia.yaml}"
CAMERA_CONFIG="${3:-}"
OUTPUT_DIR="${4:-$SCRIPT_DIR/output}"

echo "========================================"
echo " FAST-LIVO2 Python SLAM Pipeline"
echo "========================================"
echo "Bag:    $BAG_PATH"
echo "Config: $CONFIG"
echo "Camera: $CAMERA_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "========================================"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CAMERA_ARG=""
if [ -n "$CAMERA_CONFIG" ]; then
    CAMERA_ARG="--camera-config $CAMERA_CONFIG"
fi

python3 "$SCRIPT_DIR/fast_livo2.py" \
    --bag "$BAG_PATH" \
    --config "$CONFIG" \
    $CAMERA_ARG \
    --output "$OUTPUT_DIR"
