#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BOARD=${BOARD:-globi}
LOCAL_BIN=${LOCAL_BIN:-$SCRIPT_DIR/build/cvi_minimal_runner}
BOARD_BIN=${BOARD_BIN:-/tmp/cvi_minimal_runner}
LOCAL_MODEL=${LOCAL_MODEL:-$REPO_ROOT/duo-tpu/workspace/traffic_light_yolov8n_named/work/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel}
LOCAL_IMAGE=${LOCAL_IMAGE:-$REPO_ROOT/duo-tpu/workspace/tpu-mlir/regression/image/dog.jpg}
IMAGE_ON_BOARD=${IMAGE_ON_BOARD:-/tmp/$(basename "$LOCAL_IMAGE")}
MODEL_ON_BOARD=${MODEL_ON_BOARD:-/tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel}
OUTPUT_ON_BOARD=${OUTPUT_ON_BOARD:-/tmp/traffic_light_image_preprocess_out.npz}
VIS_ON_BOARD=${VIS_ON_BOARD:-${OUTPUT_ON_BOARD%.npz}_vis.jpg}
JSON_ON_BOARD=${JSON_ON_BOARD:-${OUTPUT_ON_BOARD%.npz}.json}
LOCAL_OUT_DIR=${LOCAL_OUT_DIR:-$SCRIPT_DIR/out}
LOCAL_VIS_DIR=${LOCAL_VIS_DIR:-$SCRIPT_DIR/vis}
CONF_THRES=${CONF_THRES:-0.25}
IOU_THRES=${IOU_THRES:-0.50}
MAX_DET=${MAX_DET:-100}

if [[ ! -x "$LOCAL_BIN" ]]; then
  echo "binary not found or not executable: $LOCAL_BIN" >&2
  echo "run ./build.sh first" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_MODEL" ]]; then
  echo "local model not found: $LOCAL_MODEL" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_IMAGE" ]]; then
  echo "local image not found: $LOCAL_IMAGE" >&2
  exit 1
fi

mkdir -p "$LOCAL_OUT_DIR" "$LOCAL_VIS_DIR"
scp "$LOCAL_BIN" "$LOCAL_MODEL" "$LOCAL_IMAGE" "$BOARD:/tmp/"
ssh "$BOARD" "chmod +x '$BOARD_BIN' && '$BOARD_BIN' --model '$MODEL_ON_BOARD' --image '$IMAGE_ON_BOARD' --output '$OUTPUT_ON_BOARD' --save-vis '$VIS_ON_BOARD' --save-json '$JSON_ON_BOARD' --conf '$CONF_THRES' --iou '$IOU_THRES' --max-det '$MAX_DET' --count 1"
scp "$BOARD:$OUTPUT_ON_BOARD" "$LOCAL_OUT_DIR/$(basename "$OUTPUT_ON_BOARD")"
scp "$BOARD:$VIS_ON_BOARD" "$LOCAL_VIS_DIR/$(basename "$VIS_ON_BOARD")"
scp "$BOARD:$JSON_ON_BOARD" "$LOCAL_VIS_DIR/$(basename "$JSON_ON_BOARD")"
sha256sum "$LOCAL_OUT_DIR/$(basename "$OUTPUT_ON_BOARD")" "$LOCAL_VIS_DIR/$(basename "$VIS_ON_BOARD")" "$LOCAL_VIS_DIR/$(basename "$JSON_ON_BOARD")"
