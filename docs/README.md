# 工程文档索引

## 环境与基线

- [DuoS 官方开发手册整理](duos-official-development-handbook.md)：SDK、TDL、媒体链路和板端开发入口。
- [DuoS 官方基线审计](2026-07-05-duos-official-baseline-audit.md)：项目代码与官方样例的对应关系。

## 训练与模型转换

- [交通灯 YOLOv5s 训练](2026-05-04-traffic-light-yolov5s-training.md)：数据配置、训练和模型导出记录。
- [交通灯 YOLOv5s TPU-MLIR](2026-05-04-traffic-light-yolov5s-tpu-mlir.md)：ONNX 到 `cvimodel` 的转换过程。
- [组员模型转换](2026-07-06-teammate-yolov5s-cvimodel.md)：当前交通灯模型的校准与转换说明。

## 板端部署与验证

- [板端验证与 YOLOv5u 后处理](2026-05-04-traffic-light-board-validation-and-yolov5u-note.md)：静态图 runner、输出解析和阈值验证。
- [实时 runner 与 YOLOv5s](2026-05-05-live-runner-rtsp-inference.md)：摄像头、推理和 RTSP 链路。
- [实时推理集成](2026-05-05-process-14-live-inference-integration.md)：从稳定媒体骨架接入模型推理。
- [阶段总结](2026-05-05-final-summary.md)：模型、固件、板端运行与交付状态。

## 调试复盘

- [段错误根因与稳定骨架](2026-05-05-process-12-live-runner-sigsegv-root-cause-and-stable-skeleton.md)：媒体初始化顺序和崩溃定位。
- [重启恢复与 MMF 清理](2026-05-05-process-13-live-runner-restart-recovery-and-mmf-recover.md)：异常退出后的资源恢复。
- [实时推理集成记录](2026-05-05-live-inference-integration.md)：接口拆分和集成过程。

这些文档记录特定模型、固件和 SDK 环境下的工程过程。命令中的模型名、板端目录和工具链路径需要按你的环境调整。
