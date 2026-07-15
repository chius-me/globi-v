#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$RUNNER_DIR"

fail() {
  printf '[baseline-check] FAIL: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[baseline-check] %s\n' "$*"
}

require_pattern() {
  pattern="$1"
  shift
  rg -n -- "$pattern" "$@" >/dev/null || fail "missing required pattern '$pattern' in $*"
}

reject_pattern() {
  pattern="$1"
  shift
  if rg -n -- "$pattern" "$@" >/tmp/tllr-baseline-rg.txt; then
    cat /tmp/tllr-baseline-rg.txt >&2
    fail "forbidden pattern '$pattern' found in $*"
  fi
}

log 'checking source path invariants'
require_pattern 'PIXEL_FORMAT_RGB_888_PLANAR' src/main.c src/live_inference.cpp
require_pattern 'CVI_NN_SetTensorWithAlignedFrames' src/live_inference.cpp
require_pattern 'CVI_NN_PIXEL_RGB_PLANAR' src/live_inference.cpp
require_pattern 'CVI_TDL_Service_ObjectDrawRect' src/main.c
require_pattern 'cvtdl_object_t' src/main.c
require_pattern 'CVI_TDL_Service_CreateHandle' src/main.c
require_pattern 'DUO_TDL_EXAMPLES_LIB' build.sh
require_pattern '-lcvi_tdl' build.sh
require_pattern 'BOARD_HOST=\$\{BOARD_HOST:-milkv-duo\}' scripts/deploy_live_runner.sh
require_pattern 'scp -O' scripts/deploy_live_runner.sh
require_pattern 'duos_mmf_recover\.sh' scripts/deploy_live_runner.sh

log 'checking removed experimental paths stay removed'
reject_pattern 'PIXEL_FORMAT_BGR_888_PLANAR|CVI_NN_PIXEL_BGR_PLANAR' src/main.c src/live_inference.cpp
reject_pattern 'CopyBgrPlanarFrameToRgbTensor|RefineTrafficLightClassFromInput|split-debug|yolov5s-debug' src/live_inference.cpp
reject_pattern 'draw_nv21|draw_y_|draw_uv|CVI_RGN_Destroy|cvi_region|IonFlush' src/main.c

log 'running build'
./build.sh >/tmp/tllr-baseline-build.log
tail -n 20 /tmp/tllr-baseline-build.log

log 'checking linked official service dependencies'
readelf -d build/traffic_light_live_runner | rg 'libcvi_tdl\.so|libopencv_core\.so|libcvi_ive\.so' >/dev/null ||
  fail 'expected TDL/OpenCV/IVE dynamic dependencies were not found'

log 'ok'
