# LISA 交通灯 YOLOv5s 的 TPU-MLIR 编译记录（未接板）

## 本轮目标

在不连接 DuoS 开发板的前提下，先把已经训练完成的 `best.onnx` 跑通 TPU-MLIR 流程，生成可供后续板端验证的 `cv181x` INT8 `cvimodel`。

## 前提检查

已确认以下条件满足：

- ONNX 输入文件存在：
  - `~/repo/github/globi/training/runs/traffic_light_yolov5s/weights/best.onnx`
  - 大小：`36,671,947 bytes`
- Docker 容器运行正常：
  - 容器名：`DuoTPU`
  - 镜像：`sophgo/tpuc_dev:v3.1`
- 目标 workspace 已准备：
  - `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/`
- 校准数据集可用：
  - `/workspace/tpu-mlir/regression/dataset/COCO2017/`
  - 检测到 `100` 个输入样本
- 工具链版本：
  - `SOPHGO Toolchain v1.3.228-g19ca95e9-20230921`

## 执行流程

### 1. 复制 ONNX 到 TPU workspace

- 来源：`training/runs/traffic_light_yolov5s/weights/best.onnx`
- 目标：`duo-tpu/workspace/traffic_light_yolov5s/best.onnx`

### 2. model_transform.py

执行参数：

- `--model_name traffic_light_yolov5s`
- `--model_def ../best.onnx`
- `--input_shapes [[1,3,640,640]]`
- `--mean 0.0,0.0,0.0`
- `--scale 0.0039216,0.0039216,0.0039216`
- `--keep_aspect_ratio`
- `--pixel_format rgb`
- `--test_input /workspace/tpu-mlir/regression/image/dog.jpg`
- `--test_result traffic_light_yolov5s_top_outputs.npz`
- `--mlir traffic_light_yolov5s.mlir`

### 3. calibration + model_deploy.py

执行参数：

- `run_calibration.py traffic_light_yolov5s.mlir --dataset /workspace/tpu-mlir/regression/dataset/COCO2017 --input_num 100 -o traffic_light_yolov5s_cali_table`
- `model_deploy.py --mlir traffic_light_yolov5s.mlir --quant_input --quant_output --quantize INT8 --calibration_table traffic_light_yolov5s_cali_table --chip cv181x --model traffic_light_yolov5s_cv181x_int8_sym.cvimodel`

## 关键结果

### model_transform 结果

日志显示：

- `193 compared`
- `193 passed`
- `0 failed`
- `npz compare PASSED`
- `Mlir file generated: traffic_light_yolov5s.mlir`

说明 ONNX → MLIR 转换成功，且与参考输出比对通过。

### calibration / deploy 结果

日志显示：

- `real input_num = 100`
- `prepare data from 100`
- `auto tune end, run time:161.63821840286255`
- `tpuc-opt ... --chip-assign="chip=cv181x" ... [Success]`
- `tpuc-opt ... --codegen="model_file=traffic_light_yolov5s_cv181x_int8_sym.cvimodel ..." [Success]`

说明校准、量化、编译、codegen 均已跑通。

## 生成产物

### 主产物

- MLIR：
  - `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s.mlir`
  - 大小：`69,842 bytes`
  - SHA256：`be62bf8cf925a64d74bfcda48e8b8376bbde41f9cae54e30ea228e88cf0d1904`

- Calibration table：
  - `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_cali_table`
  - 大小：`13,848 bytes`
  - SHA256：`29571205a0b13c2872765e9783359dd55c5b65f3f7d971230b8763e9b8af03ad`

- CV181x INT8 cvimodel：
  - `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_cv181x_int8_sym.cvimodel`
  - 大小：`10,366,256 bytes`
  - SHA256：`50c7795fb02960e573b6eee9b72d7b5e875d29dc54687d32bbc791f0fad78a5c`

### 中间产物

- `traffic_light_yolov5s_origin.mlir`
- `traffic_light_yolov5s_cv181x_int8_sym_tpu.mlir`
- `traffic_light_yolov5s_cv181x_int8_sym_final.mlir`
- `traffic_light_yolov5s_in_f32.npz`
- `model_transform_traffic_light_yolov5s.log`
- `model_deploy_traffic_light_yolov5s.log`

## 当前结论

1. **TPU-MLIR 主线成功跑通**：无需接板，已经把训练产物成功编译成 `cv181x` 的 INT8 `cvimodel`。
2. **模型可进入下一阶段**：后续只需要在板端做离线图片验证，再决定是否接入实时链路。
3. **当前没有发现像 YOLO26 那样的算子不支持问题**：这个 YOLOv5s 交通灯模型可以被当前 `v1.3.228` 工具链正常处理。

## 待做 / 下一步

### 接上板后

推荐直接做最小验证：

1. 把 `traffic_light_yolov5s_cv181x_int8_sym.cvimodel` 上传到 DuoS
2. 先用板端样例做离线图片推理
3. 观察输出打印、TPU 使用计数与结果是否合理
4. 再决定是否接入摄像头实时链路

### 推荐的板端验证命令

```bash
scp ~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_cv181x_int8_sym.cvimodel globi:/root/

ssh globi '
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd
/mnt/system/usr/bin/ai/sample_yolov5 \
  /root/traffic_light_yolov5s_cv181x_int8_sym.cvimodel \
  /mnt/tpu/tpu-sdk/samples/samples_extra/data/dog.jpg
'
```

## 备注

- 这次工作是 **纯主机侧 TPU-MLIR 编译**，尚未进行板端实际推理验证。
- 因为还没接板，所以当前只能确认：
  - ONNX 正常
  - MLIR 正常
  - calibration 正常
  - cvimodel 已成功生成
- 不能替代板端真实运行验证，特别是：
  - 板端样例是否正常打印检测结果
  - 实际推理耗时
  - TPU 使用情况
  - 板端输入前处理是否与训练假设一致
