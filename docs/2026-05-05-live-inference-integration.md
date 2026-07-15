# 过程 6｜Live Runner TPU 实时推理接入（Zero-Copy 前向）

## 背景

在 live skeleton（过程 3）基础上，将离线验证过的 TPU 推理逻辑接入实时视频管线。

## 实现方案

### 架构
```
Camera (VI) → VPSS Chn0 (1280×720, RTSP)
            → VPSS Chn1 (640×640, BGR planar → TPU)
```

### Zero-Copy TPU 前向
直接从 VPSS Chn1 的 BGR planar 帧获取物理地址，无需 OpenCV 拷贝：

```c
CVI_NN_SetTensorWithAlignedFrames(
    &ctx->inputs[0],
    frame_paddrs,     // {phyB, phyG, phyR}
    1,                // batch=1 (不是 plane count=3!)
    CVI_NN_PIXEL_BGR_PLANAR
);
```

**关键陷阱**: `frame_num=1`（batch size），不是 3（plane count）。传 3 会触发运行时断言。

### 性能

| 指标 | YOLOv8n_named (int8) | YOLOv5s split (fp32 output) |
|---|---|---|
| Forward time | ~117 ms | ~220 ms |
| RTSP 帧率 | ~28 fps | ~25 fps |

### RGN 硬件叠加
使用 CVI RGN (Region) 在 VPSS Chn0 叠加两个绿色方块作为调试标记（左下 + 右上），验证 VPSS→VENC→RTSP 链路完整。注意：绿方块 **不是** 检测结果。

### NV21 检测框叠加
分析线程通过 `pthread_mutex` 保护的共享缓冲区写入检测结果，RTSP 线程读取后在 NV21 Y-plane 上 CPU 绘制白色边框，调用 `CVI_SYS_IonFlushCache` 后送入编码器。

**关键陷阱**:
- 忘记 Ion Flush → 框不可见
- NV21 Y-plane stride 可能大于帧宽（对齐）
- 640×640 → 1280×720 坐标映射按比例（未考虑 center-crop 偏移）

## 多格式解码器

```
output_num==2 && bbox[4,N]+cls[nc,N] → DecodeYolov5sSplitDetections  (5月5日新增)
output_num==1 && concat[4+nc,N]      → DecodeYolov5sDetections
output_num==6 (default)              → DecodeYolov8Detections
```
