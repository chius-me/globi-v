# traffic-light-live-runner

实时交通灯检测程序：摄像头 → VPSS → RTSP 输出 + TPU 推理。

## 当前状态

**实时推理闭环已打通（过程 14，2026-05-05）：**

- 从 `VPSS_CHN1` 取 640x640 RGB planar 帧
- 通过 `CVI_NN_SetTensorWithAlignedFrames` 直接喂给 TPU（零拷贝，不需要 OpenCV 前处理）
- `CVI_NN_Forward` 实时推理，~117ms/帧
- 板端 YOLOv8 decode + NMS + 类别映射
- 控制台实时打印 `red_light / green_light / yellow_light` 结果
- RTSP 同步推流 1280x720 H.264 (~28fps)
- RTSP 检测框使用官方 `CVI_TDL_Service_ObjectDrawRect` 绘制

## 目录

- `src/main.c`：主程序（C，线程调度 / 参数解析 / middleware 管理）
- `src/live_inference.h`：推理模块 C API（模型加载 / 帧推断 / 结果输出）
- `src/live_inference.cpp`：推理模块实现（cviruntime / decode / NMS，封装自 minimal-runner）
- `build.sh`：交叉编译脚本（C + C++ 混合编译，链接 TPU SDK）
- `scripts/check_live_runner_baseline.sh`：主机侧基线检查（RGB 输入 / 官方 TDL service overlay / 构建）
- `scripts/deploy_live_runner.sh`：构建、部署到 DuoS，并可选做限时烟测
- `scripts/duos_mmf_recover.sh`：板端 MMF / VPSS 脏状态恢复脚本

## 构建

```bash
cd ~/repo/github/globi/board-runtime/traffic-light-live-runner
chmod +x build.sh
./build.sh
```

改动前后建议跑完整基线检查：

```bash
./scripts/check_live_runner_baseline.sh
```

依赖：
- 交叉工具链：`~/repo/github/globi/duo-tdl-examples/host-tools/gcc/riscv64-linux-musl-x86_64/bin`
- TDL/RTSP 参考：`~/repo/github/globi/duo-tdl-examples` 和 `~/repo/github/globi/duo-buildroot-sdk-v2/tdl_sdk`
- legacy middleware/rtsp headers：`~/repo/github/globi/cvitek-tdl-sdk-sg200x`
- TPU SDK：`~/repo/github/globi/duo-tpu/workspace/tpu-sdk`
- sample_video：`~/repo/github/globi/duo-buildroot-sdk-v2/tdl_sdk/sample_video`

## 部署和运行

```bash
# 构建 + 基线检查 + 部署到 /root/traffic-light-live
./scripts/deploy_live_runner.sh

# 部署后做 25 秒烟测
./scripts/deploy_live_runner.sh --run

# 板端手动运行
ssh milkv-duo '/root/traffic-light-live/duos_mmf_recover.sh || true; /root/traffic-light-live/run.sh'
```

默认 SSH 目标是 `milkv-duo`，默认模型是
`~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov8n_named/work/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel`。
可用环境变量覆盖，例如：

```bash
BOARD_HOST=root@192.168.42.1 CONF=0.10 INFER_EVERY=4 ./scripts/deploy_live_runner.sh --run
```

## 运行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必填) | cvimodel 路径 |
| `--rtsp-width` | 1280 | RTSP 输出宽度 |
| `--rtsp-height` | 720 | RTSP 输出高度 |
| `--analysis-width` | 640 | 推理通道宽度 |
| `--analysis-height` | 640 | 推理通道高度 |
| `--fps` | 20 | 传感器/编码器帧率提示 |
| `--rtsp-port` | 554 | RTSP 端口 |
| `--conf` | 0.25 | 置信度阈值 |
| `--iou` | 0.50 | NMS IoU 阈值 |
| `--max-det` | 100 | 每帧最大检测数 |
| `--infer-every` | 1 | 每 N 帧推理一次 |

## 预期日志

启动后：
```
[live-infer] model loaded successfully
[live-infer] middleware init complete
[live-infer] RTSP path ready: rtsp://<board-ip>:554/h264
[analysis-infer] started on VPSS grp=0 chn=1
[rtsp-output] started on VPSS grp=0 chn=0
```

有检测时：
```
[analysis-infer] frame=112 det=1 forward_ms=117.37 | green_light:0.260@53,334,60,346
[analysis-infer] *** DETECTED: green_light conf=0.260 x1y1=53,334 x2y2=60,346 ***
```

无检测时：
```
[analysis-infer] frame=4 det=0 forward_ms=117.16
```

正常退出：
```
[signal] received 2, exiting...
[rtsp-output] stopped
[analysis-infer] stopped (frames=260 infers=65 avg_forward_ms=117.4)
[live-infer] shutdown complete
```

## 恢复路径

- `SIGTERM` / `SIGINT` / `SIGHUP` → graceful cleanup → 可立即重启
- `SIGKILL` 不可捕获 → VPSS 残留 → 需先执行恢复脚本：

```bash
ssh milkv-duo '/root/traffic-light-live/duos_mmf_recover.sh'
```

## 架构说明

```
camera (GC2083 1920x1080)
  └─ VI Pipe 0
       └─ VPSS Group 0
            ├─ CHN0 (1280x720 NV21) → RTSP (H.264)
            └─ CHN1 (640x640 RGB planar) → CVI_NN_SetTensorWithAlignedFrames → TPU Forward → Decode/NMS → 控制台输出
                                         └─ TLLR_Detection → cvtdl_object_t → CVI_TDL_Service_ObjectDrawRect → RTSP overlay
```

推理模块（`live_inference.cpp`）从 `minimal-cviruntime-runner` 提取了：
- 模型加载（`CVI_NN_RegisterModel`）
- RGB planar 帧直接喂 TPU（`CVI_NN_SetTensorWithAlignedFrames`，零拷贝）
- YOLOv8 DFL decode
- 类别 sigmoid
- 按类 NMS
- 结果摘要输出
- 检测框通过官方 TDL service 叠加到 RTSP 帧

## 下一步

1. 多跑几张真实红绿灯图验证类别稳定性
2. 补演示脚本和自动化测试
3. 恢复 USB/SSH 后部署官方 service overlay 版本并验证 VLC 画框效果
