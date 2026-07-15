# LISA 交通灯 YOLOv5s 训练完成与 ONNX 导出记录

## 本轮目标

在 WSL 的 RTX 4060 Laptop GPU 上完成基于 LISA 数据集的交通灯检测模型训练，并导出可继续进入 TPU-MLIR 流程的 ONNX 文件，作为后续 DuoS / CV181x 部署基线。

## 环境确认

- 训练脚本：`~/repo/github/globi/training/train_traffic_light.py`
- 日志文件：`~/repo/github/globi/training/runs/logs/traffic_light_yolov5s_2026-05-03_23-51-21.log`
- 训练输出目录：`~/repo/github/globi/training/runs/traffic_light_yolov5s/`
- Python / CUDA：`Python 3.12.3 + torch 2.11.0+cu130`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB)`
- 训练轮数：`50 epochs`
- 总训练时长：`5.980 hours`

## 关键结果

### 最终 best.pt 指标

来自 `results.csv` 最后一轮（同时也是最佳 `mAP50-95`）：

- Best epoch: `50`
- Precision: `0.94390`
- Recall: `0.95089`
- mAP50: `0.96884`
- mAP50-95: `0.66572`
- Final train box loss: `0.74529`
- Final val box loss: `0.94604`

### 生成产物

- PyTorch 权重：`~/repo/github/globi/training/runs/traffic_light_yolov5s/weights/best.pt`
- ONNX：`~/repo/github/globi/training/runs/traffic_light_yolov5s/weights/best.onnx`
- `best.pt` 大小：`18,507,394 bytes`（日志中 optimizer stripped 后约 `18.5MB`）
- `best.onnx` 大小：`36,671,947 bytes`（日志中约 `35.0MB`）

## ONNX 导出验证

已验证 ONNX 文件可被 `onnx.load` 与 `onnx.checker.check_model` 正常加载和检查：

- Path: `~/repo/github/globi/training/runs/traffic_light_yolov5s/weights/best.onnx`
- IR version: `6`
- Opset: `11`
- Input: `images -> [1, 3, 640, 640]`
- Output: `output0 -> [1, 7, 8400]`

结论：当前 `best.onnx` 已具备继续进入 TPU-MLIR `model_transform.py` 的基本条件。

## 导出阶段异常与解释

日志中出现了 Ultralytics 自动补装依赖失败：

- 尝试安装：`onnxslim>=0.1.71`、`onnxruntime-gpu`
- 失败原因：向 `/usr/local/lib/python3.12/dist-packages/` 写入时权限不足
- 具体报错：`Permission denied (os error 13)`

同时日志还出现：

- `WARNING ⚠️ ONNX: simplifier failure: No module named 'onnxslim'`
- `ONNX: export success ✅`

解释：

这次失败影响的是 **可选的 ONNX 简化步骤**，**不影响训练完成，也不影响 ONNX 文件本身已成功导出**。因此当前不需要重训，可以直接基于现有 `best.onnx` 继续后续 TPU-MLIR 编译。若未来确实需要“简化后的 ONNX 图”，再单独处理 `onnxslim` 的安装环境即可。

## 当前推荐结论

1. **训练主线成功**：YOLOv5s 已在 LISA 数据集上完成 50 epoch 训练，指标已达到可作为部署基线的水平。
2. **导出主线成功**：`best.onnx` 已生成且通过 ONNX checker。
3. **无需重训**：当前阻塞不在训练，而在后续 TPU-MLIR 编译与板端验证。
4. **优先下一步**：直接执行 `~/repo/github/globi/training/deploy_traffic_light.sh`，将现有 ONNX 转成 `cv181x` INT8 `.cvimodel`，再上传到 DuoS 验证。

## 可直接复用的后续命令

```bash
cd ~/repo/github/globi/training
bash deploy_traffic_light.sh
```

该脚本会完成：

1. 检查 `best.onnx`
2. 复制到 `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/`
3. 在 `DuoTPU` 容器中执行 `model_transform.py`
4. 执行 calibration 与 `model_deploy.py`
5. 生成 `traffic_light_yolov5s_cv181x_int8_sym.cvimodel`
6. 上传到板端 `globi:/root/`
7. 用板载 `sample_yolov5` 进行基础验证

## 当前风险 / 待验证点

- 还未实际执行 TPU-MLIR 编译，尚未确认该交通灯模型在当前工具链上是否存在算子兼容性问题。
- 板端 `sample_yolov5` 的打印输出可能不完整；必要时要结合 `/proc/tpu/usage_profiling` 或改用更适合打印结果的样例进一步验证。
- 现阶段指标来自桌面训练验证集，不等价于板端实时链路效果，后续仍需做部署后端到端实测。

## 下一步

### 推荐主线

- 运行 `deploy_traffic_light.sh`
- 获得 `traffic_light_yolov5s_cv181x_int8_sym.cvimodel`
- 上传到 DuoS
- 用板载样例完成首次离线图片推理验证
- 再决定是否接入实时视频链路

### 如需进一步优化

- 后续可在此基线之上尝试更严格的数据清洗与类别平衡
- 若对实时性更敏感，可在部署成功后再比较 YOLOv5s 与其他轻量结构的实际板端表现
