# 过程 14｜traffic-light-live-runner 实时推理接入与闭环验证

- 日期：2026-05-05
- 状态：推理闭环已验证通过

## 目标

将 `minimal-cviruntime-runner` 里已验证的交通灯 TPU 推理逻辑（模型加载 / forward / decode / NMS）接入 `traffic-light-live-runner` 的 analysis 线程，做到：

1. 实时从 `VPSS_CHN1` 取 640x640 BGR planar 帧
2. 喂入 TPU 做 forward
3. 板端 decode + NMS 出检测框
4. 控制台实时打印 `red_light / green_light / yellow_light` 结果
5. 不影响 RTSP 推流

## 新增文件

- **`src/live_inference.h`**：推理模块 C API
  - `tllr_inference_init()`：加载 cvimodel，查询 I/O tensor
  - `tllr_inference_destroy()`：清理
  - `tllr_inference_run_frame()`：拿 `VIDEO_FRAME_INFO_S` 直接跑推理，返回检测结果
  - `tllr_class_name()`：类别 ID → 名称

- **`src/live_inference.cpp`**：推理实现
  - 模型加载：`CVI_NN_RegisterModel` → `CVI_NN_SetConfig` → `CVI_NN_GetInputOutputTensors`
  - 输入喂帧：`CVI_SYS_IonInvalidateCache` → `CVI_NN_SetTensorWithAlignedFrames`（零拷贝，BGR planar → TPU）
  - Forward：`CVI_NN_Forward`
  - Decode：DFL decode → sigmoid class logits → 按类 NMS → 坐标 clamp
  - 摘要输出：每帧生成一行 `det=N forward_ms=X | red_light:conf@xyxy | ...`

## 修改文件

### `src/main.c`

- 从 skeleton 版升级为真正推理版
- 新增 CLI 参数：`--model`（必填）、`--conf`、`--iou`、`--max-det`、`--infer-every`
- 新增 `AnalysisThreadArgs`，携带 `TLLR_InferenceContext *`
- 替换 `run_analysis_probe_thread` → `run_analysis_inference_thread`
  - 每 `infer_every` 帧调一次 `tllr_inference_run_frame`
  - 有检测时打印 `*** DETECTED: xxx ***`
  - 退出时打印统计（总帧数 / 推理次数 / 平均 forward 耗时）
- `main()` 内先初始化模型再初始化 middleware，退出时先 destroy middleware 再 destroy 模型

### `build.sh`

- 从纯 C 编译改为 C + C++ 混合编译
- C 文件用 `riscv64-unknown-linux-musl-gcc` 编译
- C++ 文件用 `riscv64-unknown-linux-musl-g++` 编译
- 链接用 g++
- 新增 TPU SDK 路径：`$REPO_ROOT/duo-tpu/workspace/tpu-sdk`
- 新增链接库：`-lcviruntime -lcvikernel -lcnpy -lz`
- 产物：`build/traffic_light_live_runner`（stripped）、`build/traffic_light_live_runner_dbg`（debug）

## 实测结果

板端 15 秒运行（GC2083 摄像头，每 4 帧推理一次）：

| 指标 | 值 |
|------|-----|
| Forward 耗时 | ~117ms/帧，波动 ±1ms |
| RTSP 帧率 | ~28fps（同时运行） |
| 推理帧数 | 65 次（15 秒内） |
| 检测结果 | frame 112 检到 `green_light conf=0.260 @ 53,334,60,346` |
| 退出 | SIGINT → graceful cleanup 正常 |

大部分帧 `det=0`——摄像头当前场景无红绿灯，符合预期。

## 架构

```
camera (GC2083 1920x1080)
  └─ VI Pipe 0
       └─ VPSS Group 0
            ├─ CHN0 (1280x720 NV21) → RTSP (H.264)
            └─ CHN1 (640x640 BGR planar)
                 → CVI_SYS_IonInvalidateCache
                 → CVI_NN_SetTensorWithAlignedFrames (zero-copy)
                 → CVI_NN_Forward (~117ms)
                 → DFL decode → sigmoid cls → NMS
                 → printf("*** DETECTED: ... ***")
```

## 过程中的坑

1. **`CVI_NN_SetTensorWithAlignedFrames` assertion 失败**
   - 根因：`frame_num` 参数传了 3（plane 数），实际应传 1（batch 帧数）
   - 板端 libcviruntime 的 assertion 是 `frame_num <= tensor->shape.dim[0]`，即帧数 ≤ batch size
   - 对 BGR planar 单帧：`frame_num=1`，`frame_paddrs` 含 3 个 plane 地址，`pixel_format=CVI_NN_PIXEL_BGR_PLANAR`

2. **VPSS grp0 被占用**
   - 前一次 SIGKILL 残留，运行恢复脚本后正常

## 下一步

1. RTSP 帧上叠检测框（OSD / draw）
2. 多场景红绿灯实测
3. 补演示脚本
