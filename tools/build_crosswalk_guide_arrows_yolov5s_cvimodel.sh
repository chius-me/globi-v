#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST_WORK="$REPO_ROOT/duo-tpu/workspace/crosswalk_guide_arrows_yolov5s"
MODEL_NAME="crosswalk_guide_arrows_yolov5s"
YOLOV5_DIR="$HOST_WORK/yolov5"
PT_SRC="/mnt/c/Users/chius/Downloads/best.pt"
CDSET_ZIP="$REPO_ROOT/training/datasets/CDSet.zip"
S2TLD_ZIP="$REPO_ROOT/training/datasets/S2TLD%EF%BC%88720x1280%EF%BC%89.zip"

mkdir -p "$HOST_WORK/work"
cp -f "$PT_SRC" "$HOST_WORK/best.pt"
cp -f "$REPO_ROOT/training/traffic_light_yolov5_export.py" "$HOST_WORK/yolov5_export.py"

cat > "$HOST_WORK/data.yaml" <<'YAML'
nc: 2
names: [crosswalk, guide_arrows]
YAML

rm -rf "$HOST_WORK/calib"
mkdir -p "$HOST_WORK/calib"

python3 - "$CDSET_ZIP" "$S2TLD_ZIP" "$HOST_WORK/calib" <<'PY'
import re
import sys
import zipfile
from pathlib import Path

sources = [
    ("cdset", Path(sys.argv[1]), 100),
    ("s2tld720", Path(sys.argv[2]), 100),
]
out_dir = Path(sys.argv[3])

for tag, zip_path, count in sources:
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        raise SystemExit(f"missing calibration zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        images = [n for n in zf.namelist() if re.search(r"\.(jpg|jpeg|png)$", n, re.I)]
        if len(images) < count:
            raise SystemExit(f"not enough calibration images in {zip_path}: {len(images)}")
        for i in range(count):
            name = images[round(i * (len(images) - 1) / (count - 1))]
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            target = out_dir / f"{tag}_{i:03d}_{safe}"
            target.write_bytes(zf.read(name))
PY

if [ ! -d "$YOLOV5_DIR/models" ]; then
  if [ -d "$REPO_ROOT/third_party/yolov5/models" ]; then
    cp -a "$REPO_ROOT/third_party/yolov5" "$YOLOV5_DIR"
  elif [ -d "$REPO_ROOT/duo-tpu/workspace/traffic_light_teammate_yolov5s/yolov5/models" ]; then
    cp -a "$REPO_ROOT/duo-tpu/workspace/traffic_light_teammate_yolov5s/yolov5" "$YOLOV5_DIR"
  else
    git clone --depth 1 https://github.com/ultralytics/yolov5.git "$YOLOV5_DIR"
  fi
fi

docker exec DuoTPU /bin/bash -lc "
set -euo pipefail
cd /workspace/crosswalk_guide_arrows_yolov5s

python3 - <<'PY'
import importlib.util
missing = [m for m in ['torch', 'onnx', 'pandas', 'seaborn', 'cv2', 'thop'] if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing python modules in DuoTPU container: ' + ', '.join(missing))
PY

cp -f best.pt yolov5/best.pt
cp -f yolov5_export.py yolov5/yolov5_export.py
cd yolov5
python3 yolov5_export.py --weights ./best.pt --img-size 640 640
cp -f best.onnx ../best.onnx
cd ..

export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}
export PYTHONPATH=\${PYTHONPATH:-}
source /workspace/tpu-mlir/envsetup.sh >/dev/null 2>&1
mkdir -p /workspace/crosswalk_guide_arrows_yolov5s/work
cd /workspace/crosswalk_guide_arrows_yolov5s/work

model_transform.py \
  --model_name $MODEL_NAME \
  --model_def ../best.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input /workspace/tpu-mlir/regression/image/dog.jpg \
  --test_result ${MODEL_NAME}_top_outputs.npz \
  --mlir ${MODEL_NAME}.mlir \
  > model_transform_${MODEL_NAME}.log 2>&1

run_calibration.py ${MODEL_NAME}.mlir \
  --dataset /workspace/crosswalk_guide_arrows_yolov5s/calib \
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
  --tolerance 0.85,0.45 \
  --model ${MODEL_NAME}_cv181x_int8_sym.cvimodel \
  > model_deploy_${MODEL_NAME}.log 2>&1
"

ls -lh \
  "$HOST_WORK/best.pt" \
  "$HOST_WORK/best.onnx" \
  "$HOST_WORK/work/${MODEL_NAME}_cv181x_int8_sym.cvimodel"
