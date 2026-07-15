# 过程 7｜YOLOv5s 板端全链路闭环总结

## 已完成的里程碑

### 模型链路
- ✅ YOLOv5s 自定义训练 (LISA 3分类, best.pt 18.5MB)
- ✅ TPU-MLIR 编译 (ONNX → INT8 .cvimodel)
- ✅ 量化问题修复 (split 两路输出, bbox + cls 独立)
- ✅ bbox 格式修复 (xywh → xyxy 转换)
- ✅ 板端离线验证 (red_light 0.699, 与 ONNX 一致)

### 解码器架构
```
output_num==2 → DecodeYolov5sSplitDetections (YOLOv5s split)
output_num==1 → DecodeYolov5sDetections   (YOLOv5s concat)
output_num==6 → DecodeYolov8Detections    (YOLOv8 legacy)
```

### Live Runner
- ✅ RTSP 实时流 29fps
- ✅ TPU 推理 ~230ms (fp32输出) / ~117ms (int8)
- ✅ RGN 绿方块调试标记
- ✅ NV21 Y-plane 白框叠加 + Ion cache flush
- ✅ 多模型格式支持

### 文档
- ✅ 过程 5: Live Runner 实时推理
- ✅ 过程 6: TPU 实时推理接入
- ✅ 过程 12: YOLOv5s 训练
- ✅ 过程 13: YOLOv5s TPU-MLIR 编译

## 已知限制

| 项目 | 说明 |
|---|---|
| 实时检测 | 室内场景无红绿灯, 需真实交通场景 |
| 低分辨率图片 | <300px 无法检测 |
| 误检噪点 | conf<0.2 时画面顶部边缘有随机误检 |
| VPSS 脏状态 | 进程异常退出后需 reboot |

## 使用命令

```bash
# 实时检测
ssh globi '
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd
/tmp/traffic_light_live_runner \
  --model /tmp/traffic_light_yolov5s_split_cv181x_int8_sym.cvimodel \
  --rtsp-port 8554 --conf 0.25
'

# 静态图片推理
ssh globi '
export LD_LIBRARY_PATH=...
/tmp/cvi_minimal_runner \
  --model /tmp/traffic_light_yolov5s_split_cv181x_int8_sym.cvimodel \
  --image /tmp/test.jpg --output /tmp/out.npz
'

# 部署文件
scp ~/repo/github/globi/board-runtime/traffic-light-live-runner/build/traffic_light_live_runner globi:/tmp/
scp ~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_split_cv181x_int8_sym.cvimodel globi:/tmp/
```
