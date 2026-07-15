#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

OFFICIAL_BUILD_TMP=
OFFICIAL_BUILD_OUTPUTS=(
  duo-tdl-examples/sample_vi_od/sample_vi_od
  duo-tdl-examples/sample_vi_od/sample_vi_od.o
  duo-tdl-examples/common/middleware_utils.o
  duo-tdl-examples/common/sample_utils.o
)

fail() {
  printf '[duos-baseline] FAIL: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[duos-baseline] %s\n' "$*"
}

require_dir() {
  dir="$1"
  [ -d "$dir" ] || fail "missing required directory: $dir"
}

print_git_summary() {
  dir="$1"
  log "repo: $dir"
  git -C "$dir" remote -v | sed 's/^/[duos-baseline]   /' | sed -n '1,4p'
  printf '[duos-baseline]   branch: '
  git -C "$dir" rev-parse --abbrev-ref HEAD
  printf '[duos-baseline]   head: '
  git -C "$dir" rev-parse --short HEAD
}

snapshot_official_build_outputs() {
  OFFICIAL_BUILD_TMP=$(mktemp -d /tmp/duos-official-build.XXXXXX)
  : >"$OFFICIAL_BUILD_TMP/present"
  for path in "${OFFICIAL_BUILD_OUTPUTS[@]}"; do
    if [ -e "$path" ]; then
      printf '%s\n' "$path" >>"$OFFICIAL_BUILD_TMP/present"
      mkdir -p "$OFFICIAL_BUILD_TMP/$(dirname "$path")"
      cp -a "$path" "$OFFICIAL_BUILD_TMP/$path"
    fi
  done
}

restore_official_build_outputs() {
  [ -n "${OFFICIAL_BUILD_TMP:-}" ] || return 0
  [ -d "$OFFICIAL_BUILD_TMP" ] || return 0

  for path in "${OFFICIAL_BUILD_OUTPUTS[@]}"; do
    if grep -Fxq "$path" "$OFFICIAL_BUILD_TMP/present"; then
      mkdir -p "$(dirname "$path")"
      cp -a "$OFFICIAL_BUILD_TMP/$path" "$path"
    else
      rm -f "$path"
    fi
  done
  rm -rf "$OFFICIAL_BUILD_TMP"
  OFFICIAL_BUILD_TMP=
}

trap restore_official_build_outputs EXIT

require_dir milkv.io
require_dir duo-buildroot-sdk-v2
require_dir duo-tdl-examples
require_dir tdl-models
require_dir tpu-mlir
require_dir board-runtime/traffic-light-live-runner
[ -f README.md ] || fail 'missing root workspace README'
[ -f docs/duos-official-development-handbook.md ] || fail 'missing reusable DuoS development handbook'
[ -f tools/diagnose_duos_link.ps1 ] || fail 'missing Windows host link diagnostic script'

log 'checking official/local repository map'
print_git_summary milkv.io
print_git_summary duo-buildroot-sdk-v2
print_git_summary duo-tdl-examples
print_git_summary tdl-models
print_git_summary tpu-mlir

log 'checking official documentation snapshots'
rg -n 'CDC-NCM|ssh root@192\.168\.42\.1' milkv.io/docs/duo/getting-started/setup.md >/dev/null ||
  fail 'USB-NCM/SSH setup notes not found in local milkv.io docs'
rg -n 'SDK V2|DuoS|duo-buildroot-sdk-v2' milkv.io/docs/duo/getting-started/buildroot-sdk.md >/dev/null ||
  fail 'Buildroot SDK V2 notes not found in local milkv.io docs'
rg -n 'sample_vi_od|CVI_TDL_Service_ObjectDrawRect' duo-tdl-examples/sample_vi_od/sample_vi_od.c >/dev/null ||
  fail 'official sample_vi_od does not contain expected service overlay path'
rg -n 'duos-official-development-handbook|check_duos_official_baseline|duo-buildroot-sdk-v2|duo-tdl-examples' README.md >/dev/null ||
  fail 'root README is missing required workspace entrypoints'
rg -n 'Official Source Map|Official TDL Application Workflow|Anti-Drift Checklist' docs/duos-official-development-handbook.md >/dev/null ||
  fail 'reusable DuoS handbook is missing required sections'

log 'rebuilding official sample_vi_od for DuoS/RISCV64'
snapshot_official_build_outputs
(
  cd duo-tdl-examples
  export TOOLCHAIN_PREFIX="$PWD/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-"
  export CC="${TOOLCHAIN_PREFIX}gcc"
  export CHIP=CV181X
  export COMMON_DIR="$PWD/common"
  export CFLAGS="-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d -O3 -DNDEBUG -I$PWD/include/system -I$PWD/include/tdl"
  export LDFLAGS="-D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -L$PWD/libs/system/musl_riscv64 -L$PWD/libs/tdl/cv181x_riscv64"
  make -C sample_vi_od -B >/tmp/duos-sample-vi-od-build.log
)
tail -n 10 /tmp/duos-sample-vi-od-build.log
restore_official_build_outputs

log 'checking traffic live runner baseline'
board-runtime/traffic-light-live-runner/scripts/check_live_runner_baseline.sh >/tmp/duos-live-runner-baseline.log
tail -n 25 /tmp/duos-live-runner-baseline.log

log 'ok'
