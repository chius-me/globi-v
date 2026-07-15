#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST_WORK="$REPO_ROOT/duo-tpu/workspace/mangdao_yolov8seg"
MODEL_NAME="mangdao_yolov8seg"
PT_SRC="$REPO_ROOT/eval-models/mangdao-seg-2026-07-07/mangdao_seg_best.pt"
ONNX_SRC="$REPO_ROOT/eval-models/mangdao-seg-2026-07-07/mangdao_seg_best.onnx"
TP_DATASET_ZIP="/mnt/c/Users/chius/Downloads/TP-Dataset.zip"

mkdir -p "$HOST_WORK/work"
cp -f "$PT_SRC" "$HOST_WORK/best.pt"
cp -f "$ONNX_SRC" "$HOST_WORK/best.onnx"

rm -rf "$HOST_WORK/calib"
mkdir -p "$HOST_WORK/calib"

python3 - "$TP_DATASET_ZIP" "$HOST_WORK/calib" <<'PY'
import re
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
count = 200

if not zip_path.is_file() or zip_path.stat().st_size == 0:
    raise SystemExit(f"missing calibration zip: {zip_path}")

with zipfile.ZipFile(zip_path) as zf:
    images = [
        n for n in zf.namelist()
        if "JPEGImages" in n and re.search(r"\.(jpg|jpeg|png)$", n, re.I)
    ]
    if len(images) < count:
        raise SystemExit(f"not enough calibration images in {zip_path}: {len(images)}")
    for i in range(count):
        name = images[round(i * (len(images) - 1) / (count - 1))]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        target = out_dir / f"tp_{i:03d}_{safe}"
        target.write_bytes(zf.read(name))
PY

docker exec DuoTPU /bin/bash -lc "
set -eo pipefail
source /workspace/tpu-mlir/envsetup.sh >/dev/null 2>&1
mkdir -p /workspace/mangdao_yolov8seg/work
cd /workspace/mangdao_yolov8seg/work

model_transform.py \
  --model_name $MODEL_NAME \
  --model_def ../best.onnx \
  --input_shapes '[[1,3,640,640]]' \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input /workspace/tpu-mlir/regression/image/dog.jpg \
  --test_result ${MODEL_NAME}_top_outputs.npz \
  --mlir ${MODEL_NAME}.mlir \
  > model_transform_${MODEL_NAME}.log 2>&1

run_calibration.py ${MODEL_NAME}.mlir \
  --dataset /workspace/mangdao_yolov8seg/calib \
  --input_num 200 \
  -o ${MODEL_NAME}_cali_table \
  > run_calibration_${MODEL_NAME}.log 2>&1

model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quant_input --quant_output \
  --quantize INT8 \
  --calibration_table ${MODEL_NAME}_cali_table \
  --chip cv181x \
  --test_input ${MODEL_NAME}_in_f32.npz \
  --test_reference ${MODEL_NAME}_top_outputs.npz \
  --tolerance 0.80,-0.50 \
  --model ${MODEL_NAME}_cv181x_int8_sym.cvimodel \
  > model_deploy_${MODEL_NAME}.log 2>&1
"

ls -lh \
  "$HOST_WORK/best.pt" \
  "$HOST_WORK/best.onnx" \
  "$HOST_WORK/work/${MODEL_NAME}_cv181x_int8_sym.cvimodel"
