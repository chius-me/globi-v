# LISA 交通灯模型板端验证记录：YOLOv5 训练命名与 Ultralytics YOLOv5u/YOLOv8 风格兼容性排查

## 本轮目标

把昨晚训练得到的交通灯检测模型继续推进到 DuoS / CV181x 板端验证，并记录为什么“训练时看起来是 YOLOv5s”，但在部署阶段又表现出更接近 Ultralytics YOLOv5u / YOLOv8 风格的兼容性问题。

## 先说结论

1. **昨晚训练时，用户选择的起点确实是 `yolov5s.pt`。**
2. **但训练实际运行在 Ultralytics 8.4.46 框架上，日志明确提示并实际下载了 `yolov5su.pt`。**
3. **因此当前 `best.pt` / `best.onnx` 虽然命名仍沿用 `traffic_light_yolov5s`，但其部署结构更接近 Ultralytics 新版 YOLOv5u / YOLOv8 风格，而不是经典 YOLOv5 仓库那条老分支。**
4. **原始 YOLOv5 路线与重新导出的 6 输出 YOLOv8-style 路线都已经成功完成主机侧 TPU-MLIR 编译，但板端 `sample_yolov5` / `sample_yolov8` 目前仍未完成对自编译交通灯模型的真正离线推理闭环。**
5. **当前最强怀疑点已经收敛到 model name / TDL metadata 兼容性，而不只是输出 shape。**

## 训练阶段到底发生了什么

### 训练脚本中的用户意图

训练脚本 `~/repo/github/globi/training/train_traffic_light.py` 中明确写的是：

- `MODEL_NAME = "yolov5s.pt"`
- `model = YOLO(MODEL_NAME)`

也就是说，**用户的训练目标确实是 YOLOv5s**。

### 训练日志中的实际运行证据

日志文件：`~/repo/github/globi/training/runs/logs/traffic_light_yolov5s_2026-05-03_23-51-21.log`

关键原文：

- `Base model: yolov5s.pt`
- `PRO TIP 💡 Replace 'model=yolov5s.pt' with new 'model=yolov5su.pt'.`
- `YOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics`
- 实际下载：`Downloading ... yolov5su.pt`
- 环境：`Ultralytics 8.4.46`
- 检测头：`ultralytics.nn.modules.head.Detect [3, 16, None, [128, 256, 512]]`
- 冻结层：`Freezing layer 'model.24.dfl.conv.weight'`

### 解释

这说明：

- 训练入口名称是 `yolov5s.pt`
- 但实际运行时，Ultralytics 8 将它引导到了新版 `yolov5su` / YOLOv5u 体系
- 该体系在导出与检测头表现上，已经更接近 YOLOv8 风格（anchor-free / DFL 风格），而不是经典 YOLOv5 老仓库常见的那类板端兼容三输出结构

因此，后续在部署排查中说“当前模型更像 YOLOv5u / YOLOv8 风格”，并不是否认训练时用的是 YOLOv5，而是在描述**训练产物落地后的结构特征**。

## 已确认的训练与主机侧编译结果

### 训练结果

来自先前已记录的训练结果：

- 训练轮数：`50 epochs`
- 最佳轮次：`epoch 50`
- Precision：`0.94390`
- Recall：`0.95089`
- mAP50：`0.96884`
- mAP50-95：`0.66572`

### 原始 ONNX 导出结果

- ONNX 路径：`~/repo/github/globi/training/runs/traffic_light_yolov5s/weights/best.onnx`
- ONNX checker：通过
- 输入：`[1, 3, 640, 640]`
- 输出：`[1, 7, 8400]`

这一点也进一步支持：当前导出的 ONNX **不像经典板端 sample_yolov5 最容易接受的传统 YOLOv5 形态**。

### 原始 TPU-MLIR 编译结果（YOLOv5 命名路线）

已完成：

- `model_transform.py`
- `run_calibration.py`
- `model_deploy.py --chip cv181x`

关键结果：

- `193 compared`
- `193 passed`
- `0 failed`
- `npz compare PASSED`

产物：

- `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_cv181x_int8_sym.cvimodel`

### 重新导出的 6 输出 YOLOv8-style 路线

为了更贴近板端 `sample_yolov8` 的预期，又做了第二条路线：

1. 用 `yolov8_export.py` 从 `best.pt` 重新导出 6 输出 ONNX
2. 再走 TPU-MLIR 编译

关键结果：

- transform 日志：
  - `168 compared`
  - `168 passed`
  - `0 failed`
  - `npz compare PASSED`
- deploy 成功生成：
  - `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5u_v8_6out/work/traffic_light_yolov5u_v8_6out_cv181x_int8_sym.cvimodel`

## 板端验证结果

### 1. 原始 YOLOv5 路线：失败

板端执行：

```bash
/mnt/system/usr/bin/ai/sample_yolov5 /root/traffic_light_yolov5s_cv181x_int8_sym.cvimodel /root/nightTraining_nightClip5_01451.jpg
```

结果：

- `open model failed 0xc0010102!`

结论：**主机侧编译成功 ≠ 板端 sample_yolov5 一定能认这个模型。**

### 2. 6 输出 YOLOv8-style 路线：仍失败

板端执行：

```bash
/mnt/system/usr/bin/ai/sample_yolov8 /tmp/traffic_light_yolov5u_v8_6out_cv181x_int8_sym.cvimodel /root/nightTraining_nightClip5_01451.jpg
```

这轮补到的更硬证据：

- `---------------------openmodel-----------------------`
- `open model failed with 0xc0010102!`
- `RC=2`

说明这不是“外层 exit 0 但内部其实成功”的假象，而是**真实在 OpenModel 阶段失败**。

### 3. 对照模型 yolo11n：正常

为了证明不是板子环境坏了，我用本地已有对照模型在同一块板子、同一个 `sample_yolov8` 上验证。

板端结果：

- `---------------------openmodel-----------------------`
- `---------------------to do detection-----------------------`
- `image read,width:1280`
- `image read,hidth:960`
- `objnum: 0`
- `boxes=[]`

结论：

- 板子的 `sample_yolov8` 本身是好的
- `LD_LIBRARY_PATH` 配置是好的
- 问题集中在**自编译交通灯模型不被 TDL / sample_yolov8 接受**

## 当前最重要的新证据：model name 可能影响板端兼容性

为了继续定位原因，提取了 `model_tool --info` 头信息。

### 交通灯 6 输出模型头部

- `Mlir Version: v1.3.228-g19ca95e9-20230921`
- `Cvimodel Version: 1.4.0`
- `traffic_light_yolov5u_v8_6out Build at 2026-05-04 12:46:16`
- `For cv181x chip ONLY`
- `CviModel Need ION Memory Size: (17.69 MB)`

### 对照 yolo11n 模型头部

- `Mlir Version: v1.3.228-g19ca95e9-20230921`
- `Cvimodel Version: 1.4.0`
- `yolo11n Build at 2026-05-03 13:51:57`
- `For cv181x chip ONLY`
- `CviModel Need ION Memory Size: (7.68 MB)`

### 解释

两者芯片类型一致，工具链版本一致，`sample_yolov8` 也能正常打开 `yolo11n`。目前最值得怀疑的差异之一就是：

- 交通灯模型名：`traffic_light_yolov5u_v8_6out`
- 对照模型名：`yolo11n`

结合此前项目文档里的经验：

- `sample_yolov8` 可以直接加载 `yolo11n`
- 已有记录指出：TDL SDK 可能通过模型名识别架构 / 路由到 `CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION`

所以当前主判断是：

## 当前自编译交通灯模型失败，不仅仅可能是输出 shape 问题，还很可能与 model name / TDL metadata 有关

## 关于“为什么会让我感觉像不是 YOLOv5 了”

这次用户明确追问：

- “我们昨天晚上训练不是 yolov5 吗？”

这句问得非常对，正确回答应该是：

### 正确说法

- **训练起点是 YOLOv5s。**
- **但在 Ultralytics 8 里，它实际落到了 yolov5su / YOLOv5u 这条更接近 YOLOv8 的实现体系。**
- **所以部署时表现得不像经典板端 sample_yolov5 最爱吃的老 YOLOv5。**

也就是说：

- 不是“训练成了 YOLOv8”
- 而是“训练入口叫 YOLOv5s，但导出后的结构表现更接近新版 Ultralytics YOLOv5u / YOLOv8-style”

## 当前状态总结

### 已完成

- 昨晚 `yolov5s` 训练主线完成
- 原始 ONNX 导出完成
- 原始 TPU-MLIR 编译完成
- 6 输出 ONNX 重导出完成
- 6 输出 TPU-MLIR 编译完成
- 板端原始 YOLOv5 路线验证完成（失败）
- 板端 6 输出 YOLOv8-style 路线验证完成（失败）
- 对照 yolo11n 路线验证完成（成功打开）
- 已将怀疑点收敛到 model name / metadata / TDL 识别

### 未完成

- 交通灯模型尚未完成真正的板端离线推理成功闭环
- 还没完成“把 model name 改成 `yolov8n` 再重编译”的验证

## 当前推荐下一步

最优先继续做的验证只有一条：

### 把 6 输出 ONNX 用 `--model_name yolov8n` 重新编译一次

然后再次上板执行：

```bash
/mnt/system/usr/bin/ai/sample_yolov8 <new_model>.cvimodel <test_image>
```

如果这次可以正常 `OpenModel`，就基本能坐实：

- 当前主要问题不是训练失败
- 不是 TPU-MLIR 根本不能编
- 而是**板端 TDL 对自定义模型名 / 元数据识别过于严格**

## 本轮补充证据

### 最新训练导出日志补充

这轮重新核对训练收尾日志后，可以把导出阶段再说得更完整一点：

- `PyTorch: starting from .../best.pt`，输出 shape 为 `(1, 7, 8400)`
- Ultralytics 在导出时尝试自动安装 `onnxslim>=0.1.71` 和 `onnxruntime-gpu`
- 自动安装失败的直接原因是：
  - `Permission denied`
  - 无法创建 `/usr/local/lib/python3.12/dist-packages/colorama`
- 但这个失败**没有阻断导出本身**，后续日志仍然给出：
  - `ONNX: export success`
  - `saved as '.../best.onnx'`

因此这一轮可以明确补充一个判断：

- 当前 `best.onnx` 是**有效导出成功**的
- 但导出环境仍有一个非致命的 Python 包安装权限问题
- 这个权限问题解释了为什么日志里会出现 `onnxslim` 缺失警告，但**不是本轮板端 OpenModel 失败的主因**

### 6 输出 cvimodel 产物指纹

当前自编译 6 输出模型产物已确认：

- 路径：`~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov5u_v8_6out/work/traffic_light_yolov5u_v8_6out_cv181x_int8_sym.cvimodel`
- 大小：`10,250,576 bytes`
- SHA256：`5f75dd892299e5eec536bd19ebbebc1eba1664108bc208e4c732874d3116a550`

这份指纹后续可以直接用于：

- 与重新命名为 `yolov8n` 后的新模型做一一对照
- 避免后面板端测试时把旧模型和新模型混淆
- 写入比赛材料 / PPT / 复现实验记录

## 当前一句话结论

**昨晚训练确实是以 `yolov5s` 为起点，但 Ultralytics 8 实际把它带到了新版 `yolov5su` / YOLOv5u 风格的实现路径；这也是为什么后续在 DuoS 板端既不像经典 YOLOv5 那样能被 `sample_yolov5` 直接接受，又还没完全满足 `sample_yolov8` 对模型名 / metadata 的兼容要求。**

## 补充验证：绕开 TDL 高层后的低层 runtime 结果

在这轮补充验证里，不再走板端 `/mnt/system/usr/bin/ai/sample_yolov8` 这条 TDL 高层检测 sample，而是直接改用板端 TPU runtime 自带的低层 runner：

- `/mnt/tpu/tpu-sdk/bin/model_runner`
- `/mnt/tpu/tpu-sdk/samples/bin/cvi_sample_model_runner`

这两条路径背后对应的是更底层的 runtime API：

- `CVI_NN_RegisterModel`
- `CVI_NN_GetInputOutputTensors`
- `CVI_NN_Forward`
- `CVI_NN_CleanupModel`

也就是说，这一轮验证的重点已经从“TDL 是否认这个 YOLO 模型”切换成了：

- **底层 runtime 能不能直接把这个 `.cvimodel` 打开**
- **打开后能不能跑一次前向**
- **能不能把输出 tensor 真正落盘导出来**

### 低层 runtime 直接打开自训练交通灯模型：成功

在板端执行：

```bash
export LD_LIBRARY_PATH=/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd:$LD_LIBRARY_PATH
/mnt/tpu/tpu-sdk/bin/model_runner \
  --model /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel
```

板端返回关键信号：

- `cvimodel's version:1.4`
- 输入张量：`images <1,3,640,640>, i8`
- 输出张量共 6 个：
  - `output0_Conv <1,64,80,80>, i8`
  - `367_Conv <1,64,40,40>, i8`
  - `374_Conv <1,64,20,20>, i8`
  - `381_Conv <1,3,80,80>, i8`
  - `388_Conv <1,3,40,40>, i8`
  - `395_Conv <1,3,20,20>, i8`
- 返回码：`0`

这一步非常关键，因为它说明：

- **这份自训练交通灯 `cvimodel` 不是“板子完全打不开”**
- **真正打不开它的是 TDL / `sample_yolov8` 那层高层封装**
- **底层 runtime 本身可以识别并注册这个模型**

### 低层 runtime 跑一次真实前向：成功

然后继续把 TPU-MLIR 工作区里生成好的输入 `npz` 上板：

- `~/repo/github/globi/duo-tpu/workspace/traffic_light_yolov8n_named/work/yolov8n_in_f32.npz`

在板端执行：

```bash
/mnt/tpu/tpu-sdk/bin/model_runner \
  --model /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel \
  --input /tmp/yolov8n_in_f32.npz \
  --output /tmp/traffic_lowlevel_out.npz
```

结果：

- 返回码：`0`
- 输出文件成功生成：`/tmp/traffic_lowlevel_out.npz`
- 输出文件大小：`563,910 bytes`
- 输出文件 SHA256：`ae927dd2fdc99e1cb724d6b545e9233fd9a4b4f10c82806eaae59122a204e14f`

另外，同样也验证了 `/mnt/tpu/tpu-sdk/samples/bin/cvi_sample_model_runner`：

```bash
/mnt/tpu/tpu-sdk/samples/bin/cvi_sample_model_runner \
  --model /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel \
  --input /tmp/yolov8n_in_f32.npz \
  --output /tmp/traffic_sample_runner_out.npz
```

结果同样成功，返回码也是 `0`。

### 对照模型 yolo11n 的低层 runner：也成功

为了避免把“runtime runner 本身没问题”误判成“自训练模型特例”，也对 `yolo11n` 做了同样的低层验证。

板端执行：

```bash
/mnt/tpu/tpu-sdk/bin/model_runner \
  --model /root/yolo11n_cv181x_int8_sym.cvimodel \
  --input /tmp/yolo11n_in_f32.npz \
  --output /tmp/yolo11n_lowlevel_out.npz
```

结果：

- 返回码：`0`
- 输出文件成功生成：`/tmp/yolo11n_lowlevel_out.npz`

另外 `cvi_sample_model_runner` 对 `yolo11n` 也成功：

- 输出文件：`/tmp/yolo11n_sample_runner_out.npz`
- 文件大小：`1,210,710 bytes`
- SHA256：`5115be5f2f211a13d9a78f96fa8af6114811b47992c8ac3c7ef49034702d42d3`

### 这轮补充验证把结论进一步收紧成什么

这轮结果把问题边界切得更清楚了：

1. **自训练交通灯模型的 `cvimodel` 文件本身是可被板端底层 runtime 打开的。**
2. **同一个模型也能在板端底层 runtime 下完成一次真实前向，并导出输出 `npz`。**
3. **所以之前 `open model failed with 0xc0010102!` 这个问题，并不代表模型文件整体损坏，也不代表 TPU runtime 根本不接受它。**
4. **真正卡住的是 TDL 高层 YOLO 模型封装 / `sample_yolov8` 这一层。**

换句话说，当前最准确的判断不再是：

- “这个自训练模型在板端跑不起来”

而应改成：

- **“这个自训练模型目前不能被板端 TDL 的 `sample_yolov8` 路线接受，但已经能被板端低层 runtime 直接加载并完成前向。”**

## 这对后续方案意味着什么

这条新证据非常重要，因为它直接改变了后续优先级：

### 已经不需要再证明“模型本体是不是死的”

这个问题已经回答了：**不是死的。**

### 现在真正的工程问题变成两条

#### 路线 A：继续追 TDL / sample_yolov8 为什么拒绝它

要回答的是：

- 是 metadata 还差什么 tag
- 是 model type 识别逻辑太严格
- 是输出命名 / postprocess 约定仍不满足 TDL 预期

#### 路线 B：直接沿着低层 runtime 做自己的最小推理程序

因为低层已经证明可行，所以这条路现在从“猜测可行”升级成了：

- **已有实证，确定可行**

如果继续沿这条路走，下一版最小程序不需要一上来就做完整检测框，而是只需要按顺序补：

1. 直接加载 `.cvimodel`
2. 喂输入
3. 调一次 `CVI_NN_Forward`
4. 把 6 个输出 tensor 打印 / 落盘
5. 再在主机侧或板端补后处理

### 当前推荐的下一步已经变化

相比之前“优先继续试改 `model_name`”，现在更推荐：

1. **优先把低层 runtime 最小推理程序固定下来**，把“可加载、可前向、可拿输出 tensor”这条链条写成一个可复用的最小样例；
2. 再决定是否继续追 TDL 高层兼容；
3. 如果比赛目标更看重“先有一个能跑的板端自定义模型基线”，那么可以暂时绕开 TDL 高层封装，把主要精力放到自写前后处理上。

## 方案 A 落地结果：最小 runtime 推理样例已做出来并跑通

这一步不再只是用 SDK 自带的 `model_runner` 当黑盒验证，而是已经把它固化成了项目内自己的最小板端程序。

新建目录：

- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/`

里面现在有 4 个核心文件：

- `src/main.cpp`
- `build.sh`
- `deploy_test_traffic_light.sh`
- `README.md`

### 这个最小程序做了什么

它只保留最必要的 runtime 流程：

1. `CVI_NN_RegisterModel`
2. `CVI_NN_SetConfig`
3. `CVI_NN_GetInputOutputTensors`
4. 把输入 `npz` 填进 input tensor
5. `CVI_NN_Forward`
6. 把 output tensor 保存成 `npz`
7. `CVI_NN_CleanupModel`

也就是说，这个程序已经把“低层 runtime 路线”从：

- SDK 自带样例可跑

变成了：

- **我们项目自己的最小可复用程序也可跑**

### 构建方式

原本先尝试了 CMake 方案，但当前宿主环境没有 `cmake` 命令，所以改成了更直接也更稳的方式：

- 直接调用 `riscv64-unknown-linux-musl-g++`
- 使用 `duo-tdl-examples` 自带交叉编译器
- 直接链接 `~/repo/github/globi/duo-tpu/workspace/tpu-sdk/lib` 下的：
  - `libcviruntime.so`
  - `libcvikernel.so`
  - `libcnpy.so`
  - `libz.so`

最终成功产物：

- 路径：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/build/cvi_minimal_runner`
- 大小：`38,704 bytes`
- SHA256：`61a6494e521fbe543cb34c7aa513334044fe25a36d1c823ffa772ad6e2f59407`

交叉编译后确认到的 ELF 关键信息：

- 架构：`RISC-V`
- 动态链接器：`/lib/ld-musl-riscv64v0p7_xthead.so.1`
- RPATH：`/mnt/tpu/tpu-sdk/lib`

这说明它在板端会优先去找板载 TPU SDK 的动态库，而不用手工再配一轮 `LD_LIBRARY_PATH`。

### 第一次上板失败原因

第一次把程序发上板后，一开始报的是：

- `Failed to open /tmp/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel`
- `invalid cvimodel file`

后来回查发现，问题不是程序本身，而是：

- 板端 `/tmp/` 里当时并没有这份模型文件
- 也没有对应输入 `yolov8n_in_f32.npz`

所以后面把 `deploy_test_traffic_light.sh` 改成：

- 先把本地 binary、model、input npz 一起 `scp` 到板端 `/tmp/`
- 再执行板端推理
- 最后把输出 `npz` 自动回收回来

### 修正后上板运行结果：成功

修正后，执行：

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
./deploy_test_traffic_light.sh
```

板端关键输出：

- `version: 1.4.0`
- `yolov8n Build at 2026-05-04 14:37:07 For platform cv181x`
- 输入：`images <1,3,640,640> fmt=i8`
- 输出 6 个 tensor：
  - `output0_Conv <1,64,80,80>`
  - `367_Conv <1,64,40,40>`
  - `374_Conv <1,64,20,20>`
  - `381_Conv <1,3,80,80>`
  - `388_Conv <1,3,40,40>`
  - `395_Conv <1,3,20,20>`
- `Forward done: count=1, total_ms=117.686, avg_ms=117.686`
- `Saved output npz: /tmp/traffic_light_minimal_runner_out.npz`

回收到本地的输出文件：

- 路径：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/traffic_light_minimal_runner_out.npz`
- 大小：`563,910 bytes`
- SHA256：`ae927dd2fdc99e1cb724d6b545e9233fd9a4b4f10c82806eaae59122a204e14f`

这个 SHA256 和之前用 SDK 自带 `model_runner` 回收出来的 `traffic_lowlevel_out.npz` 完全一致。

这点非常重要，因为它说明：

- **我们自己写的最小 runtime 程序，已经在板端复现出了和 SDK 自带 runner 一致的输出结果。**

### 这一步现在意味着什么

到这里，方案 A 已经从“想法可行”升级成了“工程上已落地”：

1. **我们已经有了项目内自己的最小板端 runtime 推理程序。**
2. **它已经能在 DuoS / CV181x 上成功加载自训练交通灯模型。**
3. **它已经能吃进 TPU-MLIR 生成的输入 `npz`，完成一次真实前向。**
4. **它输出的结果和 SDK 自带 low-level runner 对得上。**
5. **所以接下来完全可以在这个程序上继续加前处理和后处理，而不必再被 `sample_yolov8` 卡住。**

### 方案 A 的下一步最自然怎么接

现在最自然的后续就是继续在这个最小程序上迭代，而不是再回去从零开始：

1. 先加“读图 + resize/normalize/量化输入”；
2. 再加 6 个输出 tensor 的 decode；
3. 再把交通灯类别映射、阈值筛选、NMS 接上；
4. 最后再考虑封成更完整的板端离线检测程序。

## 方案 A.1 落地结果：读图片 + 前处理已经做完并跑通

在 `minimal-cviruntime-runner` 这条项目内最小 runtime 路线上，已经补上了直接读图的能力，不再只接受 TPU-MLIR 预先准备好的 `input.npz`。

当前 `src/main.cpp` 已支持：

- `--image <image.jpg>`
- `--pixel-format rgb|bgr`
- `--mean <m1,m2,m3>`
- `--scale <s1,s2,s3>`
- `--pad-value <0-255>`
- `--stretch`
- `--dump-input-f32 <file>`

这条 `--image` 路线当前实际做的是：

1. `cv::imread` 读图
2. 按输入 tensor 尺寸 resize
3. 默认保持比例做 `letterbox`
4. pad 默认填 `0`
5. BGR 转 RGB
6. 转成 NCHW float32
7. 按 `(x - mean) * scale` 做归一化
8. 再按输入 tensor 的量化参数写入 runtime input tensor

为了把它做成可复现的验证路径，还新增了一键脚本：

- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/deploy_test_image_preprocess.sh`

这条脚本会：

- 上传 `cvi_minimal_runner`
- 上传当前 `cvimodel`
- 上传测试图片
- 在板端执行 `--image` 路线
- 自动把输出 `npz` 回收到本地
- 自动输出 SHA256

当前默认验证图是：

- `~/repo/github/globi/duo-tpu/workspace/tpu-mlir/regression/image/dog.jpg`

已确认产物：

- 本地输出：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/traffic_light_image_preprocess_out.npz`
- 大小：`563,910 bytes`
- SHA256：`40669901739547d17efa030dfe681e87b7b59033a2af49905c223db1a3bb09b0`

板端重新验证时的关键运行信号包括：

- `Image preprocess: path=/tmp/dog.jpg original=768x576 resized=640x480 paste_xy=0,80 pixel_format=rgb keep_aspect_ratio=true`
- `Forward done: count=1, total_ms=118.931, avg_ms=118.931`
- `Saved output npz: /tmp/traffic_light_image_preprocess_out.npz`

这一段现在可以明确认定：

- **方案 A.1（读图片 + 前处理）已经做完，并且板端 forward 已真实跑通。**
- **现在最小程序已经不再依赖预先准备好的 `input.npz` 才能工作。**

## 方案 A.2 / A.3 落地结果：已经能把 6 个输出 tensor 解码成框，并在原图上画出来

在 A.1 跑通后，继续补了一个主机侧的后处理脚本：

- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/tools/decode_yolov8_npz.py`

这不是板端 C++ 后处理，而是一个先求稳的主机侧验证工具，用来确认：

- 板端最小程序导出的 6 个输出 tensor 是否足够恢复真实检测框
- 我们对当前模型头（Ultralytics YOLOv5u / YOLOv8-style, DFL, anchor-free）的理解是否正确

这个脚本当前做的事是：

1. 读取 `cvi_minimal_runner` 导出的 6 个输出 tensor `.npz`
2. 按每个 output tensor 的量化 `qscale` 反量化
3. 按 `reg_max=16` 做 DFL decode
4. 按 stride `8 / 16 / 32` 建 anchor points
5. 把距离分布还原成 bbox
6. 做类别 sigmoid
7. 做 NMS
8. 把框映射回原图坐标
9. 输出画框图和 JSON

当前交通灯类别映射已经接上：

- `0 -> red_light`
- `1 -> green_light`
- `2 -> yellow_light`

### 对照验证 1：dog.jpg

先用 `dog.jpg` 这张明显不应该出现交通灯检测的图做 sanity check：

- 输入：`traffic_light_image_preprocess_out.npz`
- 结果：`0` 个框

这个结果是合理的，说明 decode 逻辑没有明显跑偏到“到处乱出框”。

### 对照验证 2：LISA 验证集真实图片

随后选了一张验证集里有 3 个红灯标注的图片：

- 图片：`~/repo/github/globi/training/traffic_light_yolo/images/val/dayTest_daySequence1_00707.jpg`
- 标注文件：`~/repo/github/globi/training/traffic_light_yolo/labels/val/dayTest_daySequence1_00707.txt`
- 标注内容共 3 行，都是类别 `0`（`red_light`）

先在板端重新跑 `--image` 路线：

- 原图尺寸：`1280x960`
- 预处理日志：`resized=640x480 paste_xy=0,80`
- 板端 forward 时间：`118.109 ms`
- 回收输出：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/dayTest_daySequence1_00707_out.npz`
- SHA256：`9609006c26cfb32bcf188377cb3bc9ad60148741597311014ecd2411c3b47b54`

再用主机侧脚本解码：

- 结果：成功恢复 `3` 个框
- 类别：全部为 `red_light`
- 置信度分别约为：
  - `0.8904`
  - `0.8904`
  - `0.8279`

对应输出产物：

- 可视化图：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_decoded.jpg`
- JSON：`~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_decoded.json`

产物指纹：

- `dayTest_daySequence1_00707_decoded.jpg`
  - 大小：`138,549 bytes`
  - SHA256：`610bb727e3e1b65abbd2ca08100dd0af8132a0942e24aea4867afc9335772801`
- `dayTest_daySequence1_00707_decoded.json`
  - 大小：`1,123 bytes`
  - SHA256：`34c1af0a6dcb4baf578ef0cd966cc5e93b510b563705ff24a36f44c5d2f2dd72`

### 和主机侧 PyTorch 结果做交叉核对

为了避免“我们自己写的 decode 脚本只是碰巧看起来像对了”，还拿同一张图直接跑了主机侧 `best.pt` 的 PyTorch 推理结果做交叉核对。

PyTorch 侧对这张图也给出 `3` 个 `red_light` 检测框，坐标和置信度与我们从板端输出 `.npz` 解码出来的结果非常接近，例如：

- 我们解码：`[384.78, 431.21, 401.57, 456.02] conf=0.8904`
- PyTorch 推理：`[383.88, 430.41, 401.51, 456.47] conf=0.9047`

这说明：

- **板端最小程序导出的 6 个输出 tensor 已经足够恢复真实检测框。**
- **当前主机侧 decode / NMS 逻辑和训练时的 Ultralytics 模型输出是对得上的。**
- **方案 A.2 / A.3 已经从“理论可做”推进到了“有真实样例、有画框结果、有对照验证”的阶段。**

## 现在方案 A 走到哪一步了

截至这一轮，方案 A 已经不是只有 low-level forward 了，而是已经贯通到了：

1. 板端最小 runtime 程序可加载模型
2. 板端可直接读图并做前处理
3. 板端可完成 forward 并导出 6 个输出 tensor
4. 主机侧可把这 6 个输出 tensor 解码成检测框
5. 主机侧可在原图上画框
6. 已有至少一个 LISA 验证集真实样例完成了“板端输出 -> 解码 -> 正确画框”的闭环验证

换句话说，现在更准确的表述已经变成：

- **虽然 TDL / `sample_yolov8` 仍不接受这份自训练模型，但我们自己的方案 A 已经完成了从板端前向到真实检测框恢复的可用闭环。**

## 当前还剩什么没做完

还没完全做完的，不再是“能不能出框”，而是下面这些工程化收尾：

1. 把输入图片路径、输出图片路径、阈值参数都整成更完整的 CLI
2. 再多跑几张 `red / green / yellow` 的验证集图片，确认类别稳定性
3. 如果后面比赛展示更看重实时性，再考虑循环推理 / 视频流版本

## 方案 A 继续推进：后处理已搬到板端，单图检测全流程闭环完成

这轮继续把原来还留在主机侧 Python 的那一段后处理，真正搬进了板端 `cvi_minimal_runner` 的 C++ 程序里。

也就是说，现在板端程序已经不只是：

- 读图
- 前处理
- forward
- 导出 6 个输出 tensor

而是已经进一步做到：

- 直接读取 6 个输出 tensor
- 按各 output tensor 的量化 scale 反量化
- 做 `reg_max=16` 的 DFL decode
- 按 stride `8 / 16 / 32` 恢复框
- 做类别 sigmoid
- 做 NMS
- 把框映射回原图坐标
- 直接在板端画框并写出结果图 / JSON

### 这轮改动了什么

核心修改文件：

- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/src/main.cpp`
- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/deploy_test_image_preprocess.sh`
- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/README.md`

当前 `main.cpp` 新增的板端参数包括：

- `--save-vis <file>`
- `--save-json <file>`
- `--conf <float>`
- `--iou <float>`
- `--max-det <n>`

现在 `deploy_test_image_preprocess.sh` 也已经同步升级，会自动：

1. 上传程序 / 模型 / 图片
2. 在板端执行：读图、前处理、forward、decode、NMS、画框
3. 回收三类结果：
   - output tensor 的 `.npz`
   - 板端直接画好的 `.jpg`
   - 板端直接导出的 `.json`

### 板端真实复验：LISA 验证集图片成功直接出框

这次继续使用之前验证过的同一张 LISA 验证集图片：

- 图片：`~/repo/github/globi/training/traffic_light_yolo/images/val/dayTest_daySequence1_00707.jpg`
- 标注：`~/repo/github/globi/training/traffic_light_yolo/labels/val/dayTest_daySequence1_00707.txt`
- Ground truth：3 个 `red_light`

执行命令：

```bash
cd ~/repo/github/globi/board-runtime/minimal-cviruntime-runner
LOCAL_IMAGE=~/repo/github/globi/training/traffic_light_yolo/images/val/dayTest_daySequence1_00707.jpg \
OUTPUT_ON_BOARD=/tmp/dayTest_daySequence1_00707_out.npz \
./deploy_test_image_preprocess.sh
```

板端关键输出这次已经不只是 forward，而是明确给出了后处理结果：

- `Image preprocess: path=/tmp/dayTest_daySequence1_00707.jpg original=1280x960 resized=640x480 paste_xy=0,80 pixel_format=rgb keep_aspect_ratio=true`
- `Forward done: count=1, total_ms=118.706, avg_ms=118.706`
- `Decoded detections: 3`
- `[0] class=red_light conf=0.8904 xyxy=[384.78, 431.21, 401.57, 456.02]`
- `[1] class=red_light conf=0.8904 xyxy=[743.41, 433.56, 761.64, 460.97]`
- `[2] class=red_light conf=0.8279 xyxy=[609.55, 381.68, 626.32, 407.01]`
- `Saved annotated image: /tmp/dayTest_daySequence1_00707_out_vis.jpg`
- `Saved detections json: /tmp/dayTest_daySequence1_00707_out.json`

这说明当前已经不是“板端出 tensor，主机再算框”，而是：

- **板端已经直接把最终框算出来并画到图上了。**

### 回收到主机的板端产物

这轮从板端自动回收了 3 份产物：

- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/out/dayTest_daySequence1_00707_out.npz`
- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_out_vis.jpg`
- `~/repo/github/globi/board-runtime/minimal-cviruntime-runner/vis/dayTest_daySequence1_00707_out.json`

对应 SHA256：

- `dayTest_daySequence1_00707_out.npz`
  - `9609006c26cfb32bcf188377cb3bc9ad60148741597311014ecd2411c3b47b54`
- `dayTest_daySequence1_00707_out_vis.jpg`
  - `31505e3f1db8758d7a3c1ac8ae381d15a32583b46a3accfc321fa0c5f59fc3dc`
- `dayTest_daySequence1_00707_out.json`
  - `1704ca932bb6192302b9dfcf763f70e65055ba35b56d5030fad3e9d2ad27f2dc`

### 和主机侧 PyTorch 再次交叉核对

为了避免“板端 C++ 版 decode 看起来能跑，但数值偏了”，这轮又拿同一张图重新跑了一次主机侧 `best.pt` 结果做交叉核对。

PyTorch 给出的 3 个框仍然是：

- `0 0.9046766757965088 [383.88, 430.41, 401.51, 456.48]`
- `0 0.8873584866523743 [743.53, 433.57, 761.81, 461.31]`
- `0 0.8511225581169128 [609.05, 380.42, 626.52, 407.13]`

和板端 C++ 结果相比，坐标与置信度都保持在非常接近的范围内。也就是说：

- **板端 C++ 后处理版和之前已验证过的主机侧 decode / PyTorch 结果仍然对得上。**

## 当前准确状态更新

截至这一轮，方案 A 已经完整推进到：

1. 板端最小 runtime 程序可加载模型
2. 板端可直接读图并做前处理
3. 板端可完成 forward 并导出 6 个输出 tensor
4. 板端可直接 decode 成检测框
5. 板端可直接做 NMS
6. 板端可直接在原图上画框并输出结果图 / JSON

所以现在最准确的表述已经变成：

- **交通灯单图离线检测的完整闭环，已经能在 DuoS 板子上单程序直接跑完。**
- **主机现在只是用来收结果，不再负责核心后处理。**
