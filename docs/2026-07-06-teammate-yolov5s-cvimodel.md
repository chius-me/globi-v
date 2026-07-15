# Teammate YOLOv5s cvimodel conversion

Date: 2026-07-06

This note records the conversion of the teammate-trained traffic-light YOLOv5 checkpoint into a DuoS/CV181x `cvimodel`.

## Source model

- Source checkpoint: `组员训练的模型/best.pt`
- Source dataset config: `组员训练的模型/data.yaml`
- Model family: Ultralytics YOLOv5 checkpoint, not YOLOv8/YOLO11.
- Input size used for export and conversion: `640x640`
- Classes from `data.yaml`:
  - `0: red`
  - `1: yellow`
  - `2: green`
  - `3: off`

The class order is part of the runtime contract. The live runner must decode class id `0/1/2/3` as `red/yellow/green/off` in that exact order.

## Official conversion basis

The conversion follows the Milk-V Duo TDL SDK YOLOv5 flow:

1. Clone/use the official Ultralytics YOLOv5 repository.
2. Install YOLOv5 requirements and `onnx`.
3. Export with the Milk-V/CVITEK `yolov5_export.py` style script, which replaces YOLOv5 forward output so RISC-V-side code handles post-processing.
4. Convert ONNX to MLIR with TPU-MLIR.
5. Run INT8 calibration.
6. Deploy MLIR plus calibration table to `cvimodel` for `cv181x`.

Official page: <https://milkv.io/docs/duo/application-development/tdl-sdk/tdl-sdk-yolov5>

Local execution uses the same containerized TPU-MLIR workflow, but directly inside WSL2 Ubuntu Docker instead of depending on the Windows Docker Desktop UI.

## Local conversion script

Script:

```bash
tools/build_teammate_yolov5s_cvimodel.sh
```

The generated files are written under `duo-tpu/workspace/traffic_light_teammate_yolov5s/`, which is intentionally ignored by git.

Container/runtime:

- Docker container: `DuoTPU`
- Container image: `sophgo/tpuc_dev:v3.1`
- Bind mount: local `duo-tpu/workspace` -> container `/workspace`
- TPU-MLIR environment: `/workspace/tpu-mlir/envsetup.sh`

Calibration set:

- Source: `training/traffic_light_yolo/images/train`
- Count: first 200 images
- Container path: `/workspace/traffic_light_teammate_yolov5s/calib`

Key conversion parameters:

```bash
model_transform.py \
  --model_name traffic_light_teammate_yolov5s \
  --model_def ../best.onnx \
  --input_shapes [[1,3,640,640]] \
  --mean 0.0,0.0,0.0 \
  --scale 0.0039216,0.0039216,0.0039216 \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --mlir traffic_light_teammate_yolov5s.mlir
```

```bash
run_calibration.py traffic_light_teammate_yolov5s.mlir \
  --dataset /workspace/traffic_light_teammate_yolov5s/calib \
  --input_num 200 \
  -o traffic_light_teammate_yolov5s_cali_table
```

```bash
model_deploy.py \
  --mlir traffic_light_teammate_yolov5s.mlir \
  --quant_input --quant_output \
  --quantize INT8 \
  --calibration_table traffic_light_teammate_yolov5s_cali_table \
  --chip cv181x \
  --test_input traffic_light_teammate_yolov5s_in_f32.npz \
  --test_reference traffic_light_teammate_yolov5s_top_outputs.npz \
  --tolerance 0.85,0.45 \
  --model traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel
```

## Outputs

ONNX:

```bash
duo-tpu/workspace/traffic_light_teammate_yolov5s/best.onnx
```

Size: about 27 MB.

cvimodel:

```bash
duo-tpu/workspace/traffic_light_teammate_yolov5s/work/traffic_light_teammate_yolov5s_cv181x_int8_sym.cvimodel
```

Size: about 7.6 MB.

`model_tool --info` summary:

- MLIR version: `v1.3.228-g19ca95e9-20230921`
- cvimodel version: `1.4.0`
- Build time: `2026-07-06 09:02:15`
- Target: `cv181x chip ONLY`
- ION memory: `14.38 MB`

Verification:

- `model_transform`: `npz compare PASSED`
- `model_deploy`: 9 outputs compared, 9 passed, 0 failed
- Final `cvimodel` vs TPU MLIR simulator: `npz compare PASSED`

## Output tensor contract

This model does not have the old single-output traffic-light contract. It exports YOLOv5 split outputs for three detection scales.

Final model outputs:

| Scale | BBox tensor | Objectness tensor | Class tensor |
| --- | --- | --- | --- |
| 80x80 | `3x80x80x4` | `3x80x80x1` | `3x80x80x4` |
| 40x40 | `3x40x40x4` | `3x40x40x1` | `3x40x40x4` |
| 20x20 | `3x20x20x4` | `3x20x20x1` | `3x20x20x4` |

The class tensor's final dimension is `4`, matching `red/yellow/green/off`.

## Runtime impact

The current live runner is not ready to run this `cvimodel` correctly yet.

Required runtime changes before board-side validation:

1. Update the traffic-light class mapping to 4 classes in model order:
   - `0: red_light`
   - `1: yellow_light`
   - `2: green_light`
   - `3: off`
2. Add a YOLOv5 9-output split decoder:
   - group each scale's bbox/objectness/class tensors
   - apply YOLOv5 anchor/grid decode
   - use `sigmoid(objectness) * sigmoid(class_score)` as confidence
   - run NMS after converting boxes back to the source frame size
3. Keep the RGB input contract:
   - TPU-MLIR: `--pixel_format rgb`
   - VPSS analysis channel: `PIXEL_FORMAT_RGB_888_PLANAR`
   - cviruntime input: `CVI_NN_PIXEL_RGB_PLANAR`

Until these changes are made, deploying this new `cvimodel` into the existing live runner can produce wrong or empty detections even though the model conversion itself passed.
