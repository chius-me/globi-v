#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-$SCRIPT_DIR/build}
TOOLCHAIN_BIN=${TOOLCHAIN_BIN:-$REPO_ROOT/duo-tdl-examples/host-tools/gcc/riscv64-linux-musl-x86_64/bin}
TPU_SDK_PATH=${TPU_SDK_PATH:-$REPO_ROOT/duo-tpu/workspace/tpu-sdk}
CXX=${CXX:-$TOOLCHAIN_BIN/riscv64-unknown-linux-musl-g++}
OUT=${OUT:-$BUILD_DIR/cvi_minimal_runner}
OPENCV_INCLUDE=${OPENCV_INCLUDE:-$TPU_SDK_PATH/opencv/include}
OPENCV_LIB=${OPENCV_LIB:-$TPU_SDK_PATH/opencv/lib}

mkdir -p "$BUILD_DIR"
export PATH="$TOOLCHAIN_BIN:$PATH"

"$CXX" \
  -O2 -std=gnu++11 \
  -mcpu=c906fdv \
  -march=rv64gcv0p7_zfh_xthead -mabi=lp64d \
  -I"$TPU_SDK_PATH/include" \
  -I"$OPENCV_INCLUDE" \
  "$SCRIPT_DIR/src/main.cpp" \
  -L"$TPU_SDK_PATH/lib" \
  -Wl,--dynamic-linker=/lib/ld-musl-riscv64v0p7_xthead.so.1 \
  -Wl,-rpath,/mnt/tpu/tpu-sdk/lib \
  -Wl,-rpath,/mnt/tpu/tpu-sdk/opencv/lib \
  -Wl,-rpath,/mnt/system/lib \
  -lcviruntime -lcvikernel -lcnpy -lz -ldl \
  "$OPENCV_LIB/libopencv_imgcodecs.so.3.2.0" \
  "$OPENCV_LIB/libopencv_imgproc.so.3.2.0" \
  "$OPENCV_LIB/libopencv_core.so.3.2.0" \
  -o "$OUT"

file "$OUT"
readelf -d "$OUT" | sed -n '1,120p'
