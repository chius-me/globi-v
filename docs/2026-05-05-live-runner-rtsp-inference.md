# 过程 7｜Live Runner 实时推理 + RGN 叠加验证

## 本轮目标

在 live runner 上完成 RTSP 实时流 + TPU 推理 + 检测框叠加的闭环验证。

## 环境确认

- **板子**: Milk-V DuoS (CV181x), Buildroot 2025.05
- **摄像头**: GC2083, 1920×1080 → VPSS: 1280×720 (RTSP) + 640×640 (分析)
- **模型**: `traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel` (10.2MB, 6-output YOLOv8 format)
- **二进制**: `traffic_light_live_runner` (May 5 15:45 构建)

## 已验证功能

### RTSP 实时流 ✅
- RTSP 输出 `rtsp://<board-ip>:8554/h264`
- 帧率稳定 ~28fps (RTSP 线程)
- GC2083 sensor 初始化成功

### TPU 推理 ✅
- 模型加载成功，6 输出 tensor 格式正确:
  - reg: [1,64,80,80], [1,64,40,40], [1,64,20,20]
  - cls: [1,3,80,80], [1,3,40,40], [1,3,20,20]
- 推理耗时 ~117ms/帧 (稳定)

### RGN 硬件叠加 ✅
- 两个绿色方块 (RGN COVER_RGN) 在左下角和右上角
- 用作调试标记，确认 VPSS→RTSP 链路完整

### 解码器代码 ✅
- `live_inference.cpp` YOLOv8 解码器 (CollectHeadPairs → DecodeYolov8Detections → NMS)
- 对 6-output YOLOv8 格式正确解析
- 崩溃/错误: 无

## 当前未完全解决的问题

### 检测输出为 0

使用旧模型 `traffic_light_yolov8n_named` 运行 99 帧，全部 `det=0`，即使 confidence 降到 0.15。

**根因分析**:
1. 解码器代码正确 (CollectHeadPairs 成功匹配 6 个输出)
2. 模型推理正常 (forward ~117ms)
3. 可能原因:
   - 摄像头画面中没有红绿灯 (室内场景)
   - 旧模型检测能力较弱
   - 需要验证: 对已知包含红绿灯的静态图片跑一轮离线推理

### YOLOv5s 新模型尚未接入

新训练的 `traffic_light_yolov5s_cv181x_int8_sym.cvimodel` (9.9MB, 单输出 [1,7,8400,1]) 使用 YOLOv5 输出格式 (concat xywh + cls)，与当前 live_inference 的 YOLOv8 6-output 解码器不兼容。

需要增加 YOLOv5 解码路径:
- 输出: `[1, 7, 8400, 1]` = [x, y, w, h, cls0, cls1, cls2]
- 无需 DFL (直接 xywh)
- 每个 grid cell 的 stride 需要从 640/feature_map_size 计算

## 下一步

1. **修复解码器**: 在 `live_inference.cpp` 中增加 YOLOv5s 解码路径
2. **上传新模型**: `traffic_light_yolov5s_cv181x_int8_sym.cvimodel` → 板子
3. **重新构建部署**: 编译新的 live runner
4. **验证**: 确保 YOLOv5s 模型在实际场景中能检出红绿灯
5. **对摄像头拍红绿灯实物/图片**: 提高检出率
