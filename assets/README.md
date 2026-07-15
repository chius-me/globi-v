# 图片与图表

本目录保存 Globi-V 的项目图表和图片类演示结果，不包含视频、模型或训练数据集。PNG/JPEG 用于 GitHub 浏览，可编辑图表同时保留 Excalidraw 源文件。

## 系统架构

| 文件 | 内容 |
| --- | --- |
| `architecture/globiv-system-architecture.*` | Globi-V 系统总体架构 |
| `architecture/hardware-software-overview.*` | 硬件与软件模块关系 |
| `architecture/vision-subsystem.*` | 三类视觉感知任务与数据流 |
| `architecture/amp-dual-core-architecture.*` | Linux 与 FreeRTOS 双核协作方案 |
| `architecture/multimodal-feedback.*` | 感知融合与反馈链路设计 |

这些文件来自本地 `deliverables/图表/`，由项目组为报告和答辩绘制。

## 工程流程

| 文件 | 内容 |
| --- | --- |
| `workflow/training-pipeline.*` | 数据准备、训练、导出与验证流程 |
| `workflow/board-inference-pipeline.*` | 摄像头到 RTSP 的板端推理管线 |
| `workflow/board-deployment-loop.*` | 模型转换、部署与交付闭环 |
| `workflow/multi-format-decoder.*` | 多种 YOLO 输出格式的统一解码结构 |
| `workflow/split-decoder-comparison.*` | Split 输出解码修复前后对比 |
| `workflow/accuracy-performance.*` | 项目测试中的精度和性能对比 |

这些图表同样来自 `deliverables/图表/`。数值只对应图中注明的模型和测试环境。

## 演示结果

- `demos/traffic-light-*.{jpg,png}`：交通灯测试图和 ONNX 检测可视化，来源为本地 `demo-images/`。
- `demos/crosswalk-detection.*`：DuoS 板端人行横道测试结果及小型 JSON 输出，来源为板端 RTSP 演示交付目录。
- `demos/crosswalk-guide-result-*.jpg`：人行横道与导向箭头检测样例，来源为本地 PT/ONNX 对比评测。
- `demos/tactile-paving-*.jpg`：DuoS BF16 输出解码后的盲道分割帧，来源为本地板端视频评测结果。

演示图片是项目生成的可视化结果。部分输入画面来自项目测试资料；公开复用原始场景图前应另行核对其来源许可。
