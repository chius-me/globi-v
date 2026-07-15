#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-$SCRIPT_DIR/build}
OUT=${OUT:-$BUILD_DIR/traffic_light_live_runner}
OUT_DBG=${OUT_DBG:-$BUILD_DIR/traffic_light_live_runner_dbg}

TOOLCHAIN_BIN=${TOOLCHAIN_BIN:-$REPO_ROOT/duo-tdl-examples/host-tools/gcc/riscv64-linux-musl-x86_64/bin}
CC=${CC:-$TOOLCHAIN_BIN/riscv64-unknown-linux-musl-gcc}
CXX=${CXX:-$TOOLCHAIN_BIN/riscv64-unknown-linux-musl-g++}
STRIP=${STRIP:-$TOOLCHAIN_BIN/riscv64-unknown-linux-musl-strip}

TDL_SDK_ROOT=${TDL_SDK_ROOT:-$REPO_ROOT/cvitek-tdl-sdk-sg200x}
MW_PATH=${MW_PATH:-$TDL_SDK_ROOT/sample/3rd/middleware/v2}
RTSP_PATH=${RTSP_PATH:-$TDL_SDK_ROOT/sample/3rd/rtsp}
DUO_TDL_ROOT=${DUO_TDL_ROOT:-$REPO_ROOT/duo-buildroot-sdk-v2/tdl_sdk}
SAMPLE_VIDEO_DIR=${SAMPLE_VIDEO_DIR:-$DUO_TDL_ROOT/sample_video}
TPU_SDK_PATH=${TPU_SDK_PATH:-$REPO_ROOT/duo-tpu/workspace/tpu-sdk}
DUO_TDL_EXAMPLES_LIB=${DUO_TDL_EXAMPLES_LIB:-$REPO_ROOT/duo-tdl-examples/libs/tdl/cv181x_riscv64}

mkdir -p "$BUILD_DIR"
export PATH="$TOOLCHAIN_BIN:$PATH"

ARCH_FLAGS=(
  -mcpu=c906fdv
  -march=rv64gcv0p7_zfh_xthead
  -mabi=lp64d
)

COMMON_CFLAGS=(
  -O2 -g
  -std=gnu11
  "${ARCH_FLAGS[@]}"
  -D_MIDDLEWARE_V2_
  -DCV181X
  -ffunction-sections
  -fdata-sections
  -Wno-pointer-to-int-cast
  -Wno-format-truncation
  -fdiagnostics-color=always
  -include "$SCRIPT_DIR/src/sample_comm_compat.h"
  -I"$SCRIPT_DIR/src"
  -I"$SAMPLE_VIDEO_DIR"
  -I"$DUO_TDL_ROOT/include"
  -I"$TDL_SDK_ROOT/include"
  -I"$TDL_SDK_ROOT/include/cvi_tdl"
  -I"$RTSP_PATH/include/cvi_rtsp"
  -I"$MW_PATH/sample/common"
  -I"$MW_PATH/include/isp/cv181x"
  -I"$MW_PATH/include/linux"
  -I"$MW_PATH/include"
)

COMMON_CXXFLAGS=(
  -O2 -g
  -std=gnu++11
  "${ARCH_FLAGS[@]}"
  -D_MIDDLEWARE_V2_
  -DCV181X
  -ffunction-sections
  -fdata-sections
  -Wno-format-truncation
  -fdiagnostics-color=always
  -I"$SCRIPT_DIR/src"
  -I"$TPU_SDK_PATH/include"
  -I"$TDL_SDK_ROOT/include"
  -I"$MW_PATH/include/linux"
  -I"$MW_PATH/include"
)

LDFLAGS=(
  -Wl,--gc-sections
  -Wl,--dynamic-linker=/lib/ld-musl-riscv64v0p7_xthead.so.1
  -Wl,-rpath,/mnt/system/lib
  -Wl,-rpath,/mnt/system/usr/lib
  -Wl,-rpath,/mnt/system/usr/lib/3rd
  -Wl,-rpath,/mnt/tpu/tdl_sdk/lib
  -Wl,-rpath,/mnt/tpu/tdl_sdk/rtsp/lib
  -L"$REPO_ROOT/duo-buildroot-sdk-v2/cvi_mpi/lib"
  -L"$REPO_ROOT/duo-buildroot-sdk-v2/cvi_mpi/lib/3rd"
  -L"$MW_PATH/lib"
  -L"$MW_PATH/lib/3rd"
  -L"$DUO_TDL_EXAMPLES_LIB"
  -L"$TDL_SDK_ROOT/lib"
  -L"$TDL_SDK_ROOT/sample/3rd/opencv/lib"
  -L"$RTSP_PATH/lib"
  -L"$TPU_SDK_PATH/lib"
)

C_SOURCES=(
  "$SCRIPT_DIR/src/main.c"
  "$SAMPLE_VIDEO_DIR/middleware_utils.c"
)

CXX_SOURCES=(
  "$SCRIPT_DIR/src/live_inference.cpp"
)

LIBS=(
  -lcvi_rtsp
  -lini
  -lsns_full
  -lsample
  -lisp
  -lvdec
  -lvenc
  -lawb
  -lae
  -laf
  -lcvi_bin
  -lcvi_bin_isp
  -lmisc
  -lisp_algo
  -lsys
  -lvi
  -lvo
  -lvpss
  -lrgn
  -lcvi_ive
  -lcvi_tdl
  -lgdc
  -lopencv_core
  -lopencv_imgproc
  -lopencv_imgcodecs
  -lcviruntime
  -lcvimath
  -lcvikernel
  -lcnpy
  -lz
  -lpthread
  -latomic
  -ldl
  -lm
)

printf '[build] CC=%s\n' "$CC"
printf '[build] CXX=%s\n' "$CXX"
printf '[build] OUT=%s\n' "$OUT"
printf '[build] SAMPLE_VIDEO_DIR=%s\n' "$SAMPLE_VIDEO_DIR"
printf '[build] MW_PATH=%s\n' "$MW_PATH"
printf '[build] RTSP_PATH=%s\n' "$RTSP_PATH"
printf '[build] TPU_SDK_PATH=%s\n' "$TPU_SDK_PATH"

# Compile C sources to object files
C_OBJS=()
for src in "${C_SOURCES[@]}"; do
  obj="$BUILD_DIR/$(basename "${src%.c}.o")"
  printf '[build] CC  %s\n' "$src"
  "$CC" "${COMMON_CFLAGS[@]}" -c "$src" -o "$obj"
  C_OBJS+=("$obj")
done

# Compile C++ sources to object files
CXX_OBJS=()
for src in "${CXX_SOURCES[@]}"; do
  obj="$BUILD_DIR/$(basename "${src%.cpp}.o")"
  printf '[build] CXX %s\n' "$src"
  "$CXX" "${COMMON_CXXFLAGS[@]}" -c "$src" -o "$obj"
  CXX_OBJS+=("$obj")
done

# Link with g++
printf '[build] LINK -> %s\n' "$OUT"
"$CXX" "${ARCH_FLAGS[@]}" "${C_OBJS[@]}" "${CXX_OBJS[@]}" "${LDFLAGS[@]}" "${LIBS[@]}" -o "$OUT"

# Save a debug (unstripped) copy
cp "$OUT" "$OUT_DBG"
"$STRIP" "$OUT" || true

file "$OUT"
readelf -d "$OUT" | sed -n '1,200p'
