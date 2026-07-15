# Minimal CVI Runtime Runner

这是给 DuoS / CV181x 准备的一个**最小板端 runtime 推理样例**。

现在它支持两条输入路线：

1. `--input <input.npz>`：直接喂已经准备好的 tensor
2. `--image <image.jpg>`：程序自己读图并做前处理

## 当前能力

### 路线 1：直接喂 npz

适合：

- 对齐 TPU-MLIR 参考输入
- 验证模型本体 + runtime 是否正常
- 排除读图/前处理干扰

流程：

1. 加载 `.cvimodel`
2. 查询输入输出 tensor
3. 把 `.npz` 输入喂进去
4. 调一次 `CVI_NN_Forward`
5. 把输出保存成 `.npz`

### 路线 2：直接读图片

适合：

- 开始进入真正板端离线推理链路
- 不再依赖预先准备好的 `input.npz`
- 为后续后处理打基础

当前 `--image` 路线实现的是：

1. `cv::imread` 读图
2. 按输入 tensor 尺寸做 resize
3. 默认保持比例 `letterbox`
4. 默认 pad 值为 `0`
5. BGR 转 RGB
6. 转成 NCHW float32
7. 按 `(x - mean) * scale` 做归一化
8. 再按输入 tensor 的量化参数写入 runtime input tensor

## 目录

- `src/main.cpp`：最小推理程序
- `build.sh`：交叉编译
- `deploy_test_traffic_light.sh`：`npz` 路线一键上板验证
- `deploy_test_image_preprocess.sh`：`image` 路线一键上板验证
- `tools/decode_yolov8_npz.py`：把 6 个输出 tensor 解码成检测框，并在原图上画框
- `vis/`：保存解码后的可视化结果和 JSON

## 依赖

默认依赖项目里已经准备好的这些路径：

- 交叉编译器：`~/repo/github/globi/duo-tdl-examples/host-tools/gcc/riscv64-linux-musl-x86_64/bin`
- TPU SDK：`~/repo/github/globi/duo-tpu/workspace/tpu-sdk`
- OpenCV 头文件：`~/repo/github/globi/duo-tpu/workspace/tpu-sdk/opencv/include`
- OpenCV 动态库：`~/repo/github/globi/duo-tpu/workspace/tpu-sdk/opencv/lib`

## 构建

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
chmod +x build.sh
./build.sh
```

构建成功后产物在：

```bash
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/build/cvi_minimal_runner
```

## 路线 1：当前交通灯模型 + npz 一键上板验证

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
chmod +x deploy_test_traffic_light.sh
./deploy_test_traffic_light.sh
```

本地回收输出位置：

```bash
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/traffic_light_minimal_runner_out.npz
```

## 路线 2：当前交通灯模型 + 真实图片一键上板验证

默认测试图：

```bash
~/repo/github/globi/duo-tpu/workspace/tpu-mlir/regression/image/dog.jpg
```

直接执行：

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
chmod +x deploy_test_image_preprocess.sh
./deploy_test_image_preprocess.sh
```

本地回收输出位置：

```bash
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/traffic_light_image_preprocess_out.npz
```

## 手动运行示例

### 喂 npz

```bash
/tmp/cvi_minimal_runner \
  --model /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel \
  --input /tmp/yolov8n_in_f32.npz \
  --output /tmp/out.npz
```

### 直接读图

```bash
/tmp/cvi_minimal_runner \
  --model /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel \
  --image /tmp/dog.jpg \
  --output /tmp/out.npz
```

### 可选参数

```bash
--pixel-format rgb|bgr
--mean 0,0,0
--scale 0.0039216,0.0039216,0.0039216
--pad-value 0
--stretch
```

## 当前约定

这版 `--image` 路线是按我们当前 TPU-MLIR 配置对齐的：

- `keep_aspect_ratio = true`
- `pad_value = 0`
- `pixel_format = rgb`
- `mean = 0,0,0`
- `scale = 0.0039216,0.0039216,0.0039216`
- `channel_format = nchw`

也就是尽量对齐：

```bash
--keep_aspect_ratio
--pixel_format rgb
--mean 0,0,0
--scale 0.0039216,0.0039216,0.0039216
```

## 路线 3：板端直接后处理 + 画框

现在 `cvi_minimal_runner` 已经不只是板端 `image -> preprocess -> forward -> output npz`，还可以直接在板端完成：

1. 读取 6 个输出 tensor
2. 按 tensor 量化 scale 反量化
3. 做 YOLOv8 / YOLOv5u 风格 DFL decode
4. 做类别 sigmoid
5. 做 NMS
6. 把框映射回原图坐标
7. 直接输出画框图和 JSON

新增参数：

```bash
--save-vis <file>
--save-json <file>
--conf <float>
--iou <float>
--max-det <n>
```

一键板端验证脚本 `deploy_test_image_preprocess.sh` 也已经同步升级，现在会自动回收：

- 输出 tensor 的 `.npz`
- 板端直接画好的 `.jpg`
- 板端直接导出的 `.json`

真实验证样例：

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
LOCAL_IMAGE=~/repo/github/globi/training/traffic_light_yolo/images/val/dayTest_daySequence1_00707.jpg \
OUTPUT_ON_BOARD=/tmp/dayTest_daySequence1_00707_out.npz \
./deploy_test_image_preprocess.sh
```

板端直接输出结果：

- `Decoded detections: 3`
- 全部类别：`red_light`
- 置信度约：`0.8904 / 0.8904 / 0.8279`
- forward 时间：`118.706 ms`

回收到主机的板端产物：

```bash
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/dayTest_daySequence1_00707_out.npz
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_out_vis.jpg
~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_out.json
```

产物指纹：

- `dayTest_daySequence1_00707_out.npz`
  - SHA256：`9609006c26cfb32bcf188377cb3bc9ad60148741597311014ecd2411c3b47b54`
- `dayTest_daySequence1_00707_out_vis.jpg`
  - SHA256：`31505e3f1db8758d7a3c1ac8ae381d15a32583b46a3accfc321fa0c5f59fc3dc`
- `dayTest_daySequence1_00707_out.json`
  - SHA256：`1704ca932bb6192302b9dfcf763f70e65055ba35b56d5030fad3e9d2ad27f2dc`

这说明当前已经完成：

- **板端读图**
- **板端前处理**
- **板端 forward**
- **板端 decode**
- **板端 NMS**
- **板端画框输出**

也就是说，交通灯单图离线检测这条链路，已经从“板端算特征、主机侧画框”推进成了：

- **整条检测闭环都能在 DuoS 板子上直接跑完。**

## 下一步

这个样例故意还是保持最小化。
后面可以继续加：

- 多跑几张 `red / green / yellow` 验证集图片，确认类别稳定性
- 把 CLI 再补完整（输入图、输出目录、阈值统一整理）
- 改成循环推理 / 视频流推理
- 如果后续展示需要，再接摄像头实时版
