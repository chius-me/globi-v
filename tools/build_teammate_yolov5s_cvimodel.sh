#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST_WORK="$REPO_ROOT/duo-tpu/workspace/traffic_light_teammate_yolov5s"
MODEL_NAME="traffic_light_teammate_yolov5s"
YOLOV5_DIR="$HOST_WORK/yolov5"
S2TLD_ZIP="$REPO_ROOT/training/datasets/S2TLD%EF%BC%88720x1280%EF%BC%89.zip"

mkdir -p "$HOST_WORK/work"
cp -f "$REPO_ROOT/组员训练的模型/best.pt" "$HOST_WORK/best.pt"
cp -f "$REPO_ROOT/组员训练的模型/data.yaml" "$HOST_WORK/data.yaml"
cp -f "$REPO_ROOT/training/traffic_light_yolov5_export.py" "$HOST_WORK/yolov5_export.py"

rm -rf "$HOST_WORK/calib"
mkdir -p "$HOST_WORK/calib"
if [ ! -s "$S2TLD_ZIP" ]; then
  echo "missing S2TLD calibration zip: $S2TLD_ZIP" >&2
  exit 1
fi

unzip -Z1 "$S2TLD_ZIP" \
  | grep -Ei '\.(jpg|jpeg|png)$' \
  | python3 -c '
import sys
images = [line.strip() for line in sys.stdin if line.strip()]
if len(images) < 200:
    raise SystemExit(f"not enough S2TLD calibration images: {len(images)}")
for i in range(200):
    print(images[round(i * (len(images) - 1) / 199)])
' \
  | while IFS= read -r image; do
      out=$(printf '%s' "$image" | tr '/ ' '__')
      unzip -p "$S2TLD_ZIP" "$image" > "$HOST_WORK/calib/$out"
    done

if [ ! -d "$YOLOV5_DIR/.git" ]; then
  git clone --depth 1 https://github.com/ultralytics/yolov5.git "$YOLOV5_DIR"
fi

docker exec DuoTPU /bin/bash -lc '
set -euo pipefail
cd /workspace/traffic_light_teammate_yolov5s

python3 - <<'"'"'PY'"'"'
import importlib.util
missing = [m for m in ["torch", "onnx", "pandas", "seaborn", "cv2", "thop"] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing python modules in DuoTPU container: " + ", ".join(missing))
PY

cp -f best.pt yolov5/best.pt
cp -f yolov5_export.py yolov5/yolov5_export.py
if [ ! -f best.onnx ]; then
  cd yolov5
  python3 yolov5_export.py --weights ./best.pt --img-size 640 640
  cp -f best.onnx ../best.onnx
  cd ..
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}
source /workspace/tpu-mlir/envsetup.sh >/dev/null 2>&1
mkdir -p /workspace/traffic_light_teammate_yolov5s/work
cd /workspace/traffic_light_teammate_yolov5s/work

model_transform.py \
  --model_name traffic_light_teammate_yolov5s \
  --model_def ../best.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input /workspace/tpu-mlir/regression/image/dog.jpg \
  --test_result traffic_light_teammate_yolov5s_top_outputs.npz \
  --mlir traffic_light_teammate_yolov5s.mlir \
  > model_transform_traffic_light_teammate_yolov5s.log 2>&1

run_calibration.py traffic_light_teammate_yolov5s.mlir \
  --dataset /workspace/traffic_light_teammate_yolov5s/calib \
  --input_num 200 \
  -o traffic_light_teammate_yolov5s_cali_table \
  > run_calibration_traffic_light_teammate_yolov5s.log 2>&1

model_deploy.py \
  --mlir traffic_light_teammate_yolov5s.mlir \
  --quant_input --quant_output \
  --quantize INT8 \
  --calibration_table traffic_light_teammate_yolov5s_cali_table \
  --chip cv181x \
  --test_input traffic_light_teammate_yolov5s_in_f32.npz \
  --test_reference traffic_light_teammate_yolov5s_top_outputs.npz \
  --tolerance 0.85,0.45 \
  --model traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel \
  > model_deploy_traffic_light_teammate_yolov5s.log 2>&1
'

ls -lh "$HOST_WORK/best.onnx" "$HOST_WORK/work/traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel"
