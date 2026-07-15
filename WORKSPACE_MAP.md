# Globi-V 公开仓库地图

## 源码

| 路径 | 内容 |
| --- | --- |
| `board-runtime/minimal-cviruntime-runner/` | 静态图片模型加载、推理、解码和结果验证 |
| `board-runtime/traffic-light-live-runner/` | 摄像头、VPSS、TPU 推理和 RTSP 实时链路 |
| `camera_capture/` | 早期摄像头采集实验，供媒体链路调试参考 |
| `training/` | 数据转换、训练、导出脚本和 YAML 配置示例 |
| `tools/` | 模型转换、图像/视频离线评测、结果标注和设备诊断 |

## 文档与展示

| 路径 | 内容 |
| --- | --- |
| `README.md` | 项目入口、依赖、常用命令和成果预览 |
| `DUOS_BASELINE.md` | DuoS 开发基线和上游依赖关系 |
| `docs/README.md` | 按主题整理的工程文档索引 |
| `assets/architecture/` | 系统架构图和可编辑 Excalidraw 图源 |
| `assets/workflow/` | 训练、转换、板端部署与调试流程图 |
| `assets/demos/` | 项目生成的检测、分割和板端验证图片 |

## 外部依赖

以下内容不会进入本仓库：

- Milk-V Duo Buildroot SDK V2；
- `duo-tdl-examples`、CVITEK TDL SDK 和 TPU-MLIR；
- 交叉编译工具链和预编译动态库；
- 模型权重、转换中间文件和训练数据集。

你可以把这些依赖放在仓库外，并通过脚本参数或环境变量指向实际路径。不要将 SDK、虚拟环境或模型复制进 Git 工作树。

## 本地生成目录

训练、转换或评测时可以在本地创建 `datasets/`、`runs/`、`exports/`、`work/`、`out/` 和 `outputs/`。根目录 `.gitignore` 会排除这些内容以及模型、视频、缓存和压缩归档。
