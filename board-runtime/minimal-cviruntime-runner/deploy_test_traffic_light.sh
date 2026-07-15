#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BOARD=${BOARD:-globi}
LOCAL_BIN=${LOCAL_BIN:-$SCRIPT_DIR/build/cvi_minimal_runner}
BOARD_BIN=${BOARD_BIN:-/tmp/cvi_minimal_runner}
LOCAL_MODEL=${LOCAL_MODEL:-$REPO_ROOT/duo-tpu/workspace/traffic_light_yolov8n_named/work/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel}
LOCAL_INPUT=${LOCAL_INPUT:-$REPO_ROOT/duo-tpu/workspace/traffic_light_yolov8n_named/work/yolov8n_in_f32.npz}
MODEL_ON_BOARD=${MODEL_ON_BOARD:-/tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel}
INPUT_ON_BOARD=${INPUT_ON_BOARD:-/tmp/yolov8n_in_f32.npz}
OUTPUT_ON_BOARD=${OUTPUT_ON_BOARD:-/tmp/traffic_light_minimal_runner_out.npz}
LOCAL_OUT_DIR=${LOCAL_OUT_DIR:-$SCRIPT_DIR/out}

if [[ ! -x "$LOCAL_BIN" ]]; then
  echo "binary not found or not executable: $LOCAL_BIN" >&2
  echo "run ./build.sh first" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_MODEL" ]]; then
  echo "local model not found: $LOCAL_MODEL" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_INPUT" ]]; then
  echo "local input npz not found: $LOCAL_INPUT" >&2
  exit 1
fi

mkdir -p "$LOCAL_OUT_DIR"
scp "$LOCAL_BIN" "$LOCAL_MODEL" "$LOCAL_INPUT" "$BOARD:/tmp/"
ssh "$BOARD" "chmod +x '$BOARD_BIN' && '$BOARD_BIN' --model '$MODEL_ON_BOARD' --input '$INPUT_ON_BOARD' --output '$OUTPUT_ON_BOARD' --count 1"
scp "$BOARD:$OUTPUT_ON_BOARD" "$LOCAL_OUT_DIR/$(basename "$OUTPUT_ON_BOARD")"

sha256sum "$LOCAL_OUT_DIR/$(basename "$OUTPUT_ON_BOARD")"
