#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNER_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$RUNNER_DIR/../.." && pwd)

BOARD_HOST=${BOARD_HOST:-milkv-duo}
BOARD_DIR=${BOARD_DIR:-/root/traffic-light-live}
MODEL_LOCAL=${MODEL_LOCAL:-$REPO_ROOT/duo-tpu/workspace/traffic_light_teammate_yolov5s/work/traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel}
MODEL_REMOTE=${MODEL_REMOTE:-$BOARD_DIR/traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel}
CONF=${CONF:-0.10}
IOU=${IOU:-0.50}
INFER_EVERY=${INFER_EVERY:-4}
SMOKE_SECONDS=${SMOKE_SECONDS:-25}
RUN_SMOKE=0
SKIP_CHECK=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build, deploy, and optionally smoke-test traffic_light_live_runner on DuoS.

Options:
  --host <ssh-host>       SSH host alias or address (default: $BOARD_HOST)
  --board-dir <path>      Board install directory (default: $BOARD_DIR)
  --model <path>          Local cvimodel path (default: $MODEL_LOCAL)
  --conf <float>          Runtime confidence threshold (default: $CONF)
  --iou <float>           Runtime NMS IoU threshold (default: $IOU)
  --infer-every <n>       Runtime inference stride (default: $INFER_EVERY)
  --smoke-seconds <n>     Smoke test duration when --run is set (default: $SMOKE_SECONDS)
  --run                   Start a bounded smoke test after deploy
  --skip-check            Skip local baseline check before deploy
  --help                  Show this message

Environment overrides use the same uppercase variable names.
EOF
}

fail() {
  printf '[deploy-live-runner] FAIL: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[deploy-live-runner] %s\n' "$*"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      [ "$#" -ge 2 ] || fail 'missing value after --host'
      BOARD_HOST="$2"
      shift 2
      ;;
    --board-dir)
      [ "$#" -ge 2 ] || fail 'missing value after --board-dir'
      BOARD_DIR="$2"
      MODEL_REMOTE="$BOARD_DIR/$(basename "$MODEL_LOCAL")"
      shift 2
      ;;
    --model)
      [ "$#" -ge 2 ] || fail 'missing value after --model'
      MODEL_LOCAL="$2"
      MODEL_REMOTE="$BOARD_DIR/$(basename "$MODEL_LOCAL")"
      shift 2
      ;;
    --conf)
      [ "$#" -ge 2 ] || fail 'missing value after --conf'
      CONF="$2"
      shift 2
      ;;
    --iou)
      [ "$#" -ge 2 ] || fail 'missing value after --iou'
      IOU="$2"
      shift 2
      ;;
    --infer-every)
      [ "$#" -ge 2 ] || fail 'missing value after --infer-every'
      INFER_EVERY="$2"
      shift 2
      ;;
    --smoke-seconds)
      [ "$#" -ge 2 ] || fail 'missing value after --smoke-seconds'
      SMOKE_SECONDS="$2"
      shift 2
      ;;
    --run)
      RUN_SMOKE=1
      shift
      ;;
    --skip-check)
      SKIP_CHECK=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[ -f "$MODEL_LOCAL" ] || fail "model not found: $MODEL_LOCAL"

if [ "$SKIP_CHECK" -eq 0 ]; then
  log 'running local baseline check'
  "$SCRIPT_DIR/check_live_runner_baseline.sh"
else
  log 'skipping local baseline check'
  "$RUNNER_DIR/build.sh"
fi

BIN_LOCAL="$RUNNER_DIR/build/traffic_light_live_runner"
[ -f "$BIN_LOCAL" ] || fail "binary not found after build: $BIN_LOCAL"

log "creating board directory: $BOARD_HOST:$BOARD_DIR"
ssh "$BOARD_HOST" "mkdir -p '$BOARD_DIR'"

log 'copying binary, model, and recovery script'
scp -O "$BIN_LOCAL" "$BOARD_HOST:$BOARD_DIR/traffic_light_live_runner"
scp -O "$MODEL_LOCAL" "$BOARD_HOST:$MODEL_REMOTE"
scp -O "$SCRIPT_DIR/duos_mmf_recover.sh" "$BOARD_HOST:$BOARD_DIR/duos_mmf_recover.sh"

log 'installing board run.sh'
ssh "$BOARD_HOST" "cat > '$BOARD_DIR/run.sh' <<'REMOTE_EOF'
#!/bin/sh
set -eu
cd '$BOARD_DIR'
exec ./traffic_light_live_runner \
  --model '$MODEL_REMOTE' \
  --conf '$CONF' \
  --iou '$IOU' \
  --infer-every '$INFER_EVERY'
REMOTE_EOF
chmod +x '$BOARD_DIR/traffic_light_live_runner' '$BOARD_DIR/duos_mmf_recover.sh' '$BOARD_DIR/run.sh'"

log "deployed to $BOARD_HOST:$BOARD_DIR"
log "RTSP URL after run: rtsp://192.168.42.1/h264"

if [ "$RUN_SMOKE" -eq 1 ]; then
  log "running $SMOKE_SECONDS second smoke test"
  ssh "$BOARD_HOST" "sh '$BOARD_DIR/duos_mmf_recover.sh' >/dev/null 2>&1 || true
cd '$BOARD_DIR'
./run.sh &
pid=\$!
sleep '$SMOKE_SECONDS'
kill -TERM \$pid >/dev/null 2>&1 || true
wait \$pid || true"
fi
