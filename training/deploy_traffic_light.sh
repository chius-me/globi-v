#!/bin/bash
# Deploy trained traffic light model to DuoS board
# Run AFTER training completes (train_traffic_light.py produces best.onnx)
set -eu

MODEL_NAME="traffic_light_yolov5s"
WORK_DIR="$HOME/repo/github/globi/duo-tpu/workspace/${MODEL_NAME}"
TRAINED_ONNX="$HOME/repo/github/globi/training/runs/${MODEL_NAME}/weights/best.onnx"

echo "=== Step 1: Check ONNX ==="
if [ ! -f "$TRAINED_ONNX" ]; then
  echo "ERROR: $TRAINED_ONNX not found. Run train_traffic_light.py first."
  exit 1
fi
ls -lh "$TRAINED_ONNX"

echo ""
echo "=== Step 2: Copy ONNX to TPU workspace ==="
mkdir -p "$WORK_DIR/work"
cp "$TRAINED_ONNX" "$WORK_DIR/"

echo ""
echo "=== Step 3: model_transform (ONNX → MLIR) ==="
docker exec DuoTPU /bin/bash -lc "
cd /workspace && source ./tpu-mlir/envsetup.sh && cd /workspace/${MODEL_NAME}/work && \
model_transform.py \
  --model_name ${MODEL_NAME} \
  --model_def ../best.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --test_input /workspace/tpu-mlir/regression/image/dog.jpg \
  --test_result ${MODEL_NAME}_top_outputs.npz \
  --mlir ${MODEL_NAME}.mlir
"

echo ""
echo "=== Step 4: Calibration + Deploy (MLIR → .cvimodel) ==="
docker exec DuoTPU /bin/bash -lc "
cd /workspace && source ./tpu-mlir/envsetup.sh && cd /workspace/${MODEL_NAME}/work && \
run_calibration.py ${MODEL_NAME}.mlir \
  --dataset /workspace/tpu-mlir/regression/dataset/COCO2017 \
  --input_num 100 \
  -o ${MODEL_NAME}_cali_table && \
model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quant_input --quant_output \
  --quantize INT8 \
  --calibration_table ${MODEL_NAME}_cali_table \
  --chip cv181x \
  --model ${MODEL_NAME}_cv181x_int8_sym.cvimodel
"

echo ""
echo "=== Step 5: Upload to board ==="
CVIMODEL="$WORK_DIR/work/${MODEL_NAME}_cv181x_int8_sym.cvimodel"
ls -lh "$CVIMODEL"
scp "$CVIMODEL" globi:/root/

echo ""
echo "=== Step 6: Verify on board ==="
ssh globi "
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd
/mnt/system/usr/bin/ai/sample_yolov5 \
  /root/${MODEL_NAME}_cv181x_int8_sym.cvimodel \
  /mnt/tpu/tpu-sdk/samples/samples_extra/data/dog.jpg
"

echo ""
echo "=== DONE ==="
echo "Model: globi:/root/${MODEL_NAME}_cv181x_int8_sym.cvimodel"
echo "To benchmark: /mnt/data/yolo_bench.sh  (or add this model manually)"
