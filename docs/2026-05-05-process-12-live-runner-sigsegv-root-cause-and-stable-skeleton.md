# 过程 12｜traffic-light-live-runner 早期崩溃根因定位与 live skeleton 稳定跑通

## 本轮目标

在 `过程 11` 已经明确“应当用稳定的视频骨架承载自有交通灯 runtime 主线”的前提下，继续把项目从路线规划推进到真实落地：

1. 让 `~/repo/github/globi/board-runtime/traffic-light-live-runner` 能在宿主机成功构建并上板；
2. 精确定位板端启动后 middleware early-init 阶段的 `SIGSEGV`；
3. 产出一个可验证的 live skeleton 结果，而不是只停在静态代码分析。

## 环境与对象

- 主工作区：`~/repo/github/globi`
- 当前主程序：`board-runtime/traffic-light-live-runner`
- 目标板：Milk-V DuoS / CV181x
- 连接方式：`ssh globi`
- 板端关键依赖目录：
  - `/mnt/system/usr/lib`
  - `/mnt/system/usr/lib/3rd`
  - `/mnt/tpu/tdl_sdk/lib`
  - `/mnt/tpu/tdl_sdk/rtsp/lib`

## 这轮实际解决了什么

### 1. 构建链路已经不是阻塞点

本轮已经把 `traffic-light-live-runner` 从“构建失败”推进到“可稳定构建、可上板执行”。

已确认的关键修正包括：

- 运行时 `rpath` 指向板端真实存在的目录：
  - `/mnt/system/usr/lib`
  - `/mnt/system/usr/lib/3rd`
  - `/mnt/tpu/tdl_sdk/lib`
  - `/mnt/tpu/tdl_sdk/rtsp/lib`
- 链接优先级改为以 `duo-buildroot-sdk-v2/cvi_mpi/lib` 为主；
- 去掉当前板端并不存在或不匹配的库依赖：
  - `-lldc`
  - `-lvpu`

这一步之后，runner 已经可以：

- 本地构建成功；
- `scp` 上板；
- `--help` 基础运行成功；
- 进入真正的 live 初始化路径。

## 2. 原始阻塞点：板端 early-init `SIGSEGV`

最开始的 live 启动表现是：

```text
[live-skeleton] starting with RTSP=1280x720, analysis=640x640, fps=20, port=554
```

随后立即在 middleware early-init 阶段崩溃。

此前从 `dmesg` 抓到的关键证据是：

```text
unhandled signal 11 code 0x1 at 0x0000000000000000
badaddr: 0000000000000000
cause: 000000000000000c
```

这高度指向空指针调用 / 解引用，而不是普通的参数失败返回。

## 3. 这次真正定位到的根因：`sample_comm.h` 头文件体系混用导致 ABI / 结构体布局不一致

仓库里同时存在两套 `sample_comm.h`：

1. `cvitek-tdl-sdk-sg200x/sample/3rd/middleware/v2/include/sample_comm.h`
2. `cvitek-tdl-sdk-sg200x/sample/3rd/middleware/v2/sample/common/sample_comm.h`

它们**不是同一个文件**，sha256 也不同。

更关键的是：

- `include/sample_comm.h` 中的 `SAMPLE_INI_CFG_S` 较短；
- `sample/common/sample_comm.h` 中的 `SAMPLE_INI_CFG_S` 包含更多字段，例如：
  - `u8MuxDev[VI_MAX_DEV_NUM]`
  - `u8AttachDev[VI_MAX_DEV_NUM]`
  - `s16SwitchGpio[VI_MAX_DEV_NUM]`
  - `u8SwitchPol[VI_MAX_DEV_NUM]`

而板端实际执行到的 `libsample.so` 中：

- `SAMPLE_COMM_VI_ParseIni`
- `SAMPLE_COMM_VI_IniToViCfg`

会按**更完整的结构体布局**去读这些内容。

这意味着：

> 如果编译 `traffic-light-live-runner` 时拿到的是较短版本的 `sample_comm.h`，那么传给 `libsample.so` 的结构体布局就会和实际库期望不一致，最终在 `SAMPLE_TDL_Get_VI_Config -> SAMPLE_COMM_VI_ParseIni / SAMPLE_COMM_VI_IniToViCfg` 这一段触发未定义行为，表现成板端 early-init `SIGSEGV`。

## 4. 这轮修复动作

修复方式不是“改业务逻辑”，而是先修正 **include 顺序**，让编译真正对齐到 `sample/common/sample_comm.h`：

- 把 `build.sh` 里的头文件顺序调整为：
  - `RTSP` 头
  - `MW_PATH/sample/common`
  - `MW_PATH/include/isp/cv181x`
  - `MW_PATH/include/linux`
  - `MW_PATH/include`

修正后，预处理结果已验证：

- `cvi_comm.h` 仍来自 `duo-buildroot-sdk-v2/tdl_sdk/include/cvi_comm.h`
- `sample_comm.h` 已切换为 `cvitek-tdl-sdk-sg200x/sample/3rd/middleware/v2/sample/common/sample_comm.h`

这一步是本轮最关键的实质性修复。

## 5. 修复后的板端验证结果：live skeleton 已真实跑起来

为了避免再靠猜测，这轮直接使用了：

- 未 strip 调试版二进制；
- 板端 `gdb` 断点；
- 对 `SAMPLE_TDL_Get_VI_Config`、`SAMPLE_COMM_VI_ParseIni`、`SAMPLE_COMM_VI_IniToViCfg` 的逐步验证。

### 关键验证链路

`gdb` 下已经确认程序依次成功走过：

1. `SAMPLE_TDL_Get_VI_Config`
2. `SAMPLE_COMM_VI_ParseIni`
3. `SAMPLE_COMM_VI_IniToViCfg`
4. `SAMPLE_TDL_Init_WM`
5. RTSP / analysis 线程启动

并且不再出现原来的 early-init 空指针崩溃。

### 已拿到的稳定运行证据

板端日志已出现：

```text
[live-skeleton] middleware init complete
[live-skeleton] RTSP path ready: rtsp://<board-ip>:554/h264
[live-skeleton] analysis channel is already running at 640x640 BGR planar for later inference
[rtsp-output] started on VPSS grp=0 chn=0
[analysis-probe] started on VPSS grp=0 chn=1
```

并且持续打印稳定帧流日志，例如：

```text
[analysis-probe] frame=120 size=640x640 fmt=3 ... avg_fps=25.27
[rtsp-output] forwarded 120 frames, avg_fps=25.23
...
[rtsp-output] forwarded 16680 frames, avg_fps≈29.96
```

这说明当前版本已经不只是“能启动”，而是已经实现：

- 摄像头采集；
- VI / ISP / VPSS 初始化；
- RTSP 路径启动；
- analysis probe 连续取帧；
- 长时间稳定跑到约 `29.9 FPS`。

## 当前新状态：原来的 `SIGSEGV` 已不是主阻塞

目前项目状态已经从：

- **“live runner 一启动就 early-init 崩溃”**

推进到了：

- **“live skeleton 已被真实跑通，且能持续转发 RTSP 与 analysis 帧流”**

这意味着主线已经进入下一阶段，而不是还卡在最早的崩溃定位。

## 当前剩余问题

### 1. 强制中断后 MMF / VPSS 可能残留脏状态

由于本轮有过 `gdb` / 外部超时强杀等调试动作，后续立即重跑时出现了：

```text
[SAMPLE_COMM_VPSS_Init]-28: CVI_VPSS_CreateGrp(grp:0) failed with 0xc0068004!
```

板端 `dmesg` 对应证据：

```text
vpss_create_grp:3331(): Grp(0) is occupied
```

所以当前剩余问题已经不是“程序逻辑仍然会早崩”，而是：

> **在异常终止 / 调试中断之后，VPSS group 0 可能残留占用，需要一个稳定的恢复路径。**

### 2. 还需要补一条“可重复重启”的收尾路径

后面最应该补的是二选一：

1. **正常退出验证**
   - 确认 `SIGINT/SIGTERM -> pthread_join -> SAMPLE_TDL_Destroy_MW` 能稳定释放资源；
2. **恢复脚本 / 恢复 helper**
   - 在板端增加一个明确的 MMF / VPSS 恢复动作，避免调试后只能靠重启板子清状态。

## 和全部文档对齐后的整体项目进度判断

把目前已有文档串起来看，整个项目已经完成了以下主线推进：

### 已完成阶段

1. **过程 1~2：训练与 TPU-MLIR 编译链路打通**
   - 训练、ONNX、`cvimodel` 产物都已完成；

2. **过程 3~5：板端 low-level runtime 与单图闭环完成**
   - 模型可在 DuoS / CV181x 上加载、前向、后处理、画框；

3. **过程 6~9：多图稳定性验证与参数固化完成**
   - 红 / 绿 / 黄三类板端抽检已做；
   - 默认参数已经收敛到可用版本；
   - yellow 难例问题被单独识别出来；

4. **过程 10：项目从单任务扩到更完整感知路线的规划完成**

5. **过程 11：实时视频流主线路线规划完成**
   - 明确应该走“稳定 live 视频骨架 + 自己的交通灯 runtime”这条线；

6. **过程 12（本轮）：live skeleton 真正落地并跑通**
   - 构建打通；
   - 原始 `SIGSEGV` 根因定位；
   - live skeleton 已验证可连续跑流。

## 当前项目所处阶段

如果用一句话概括当前阶段：

> **项目已经从“单图交通灯检测原型”推进到“实时视频 live skeleton 已验证跑通，正在补稳定重启 / 恢复机制，并准备接入真正的交通灯推理循环”。**

换成更工程化的说法就是：

- **离线检测闭环：完成**
- **板端多图稳定性验证：完成**
- **实时视频骨架：完成第一版可运行验证**
- **实时交通灯 inference 真正接入：下一主任务**
- **异常退出后的恢复与可重复重启：当前小阻塞**

## 下一步建议

下一步不该回头再做大范围路线争论，而应该直接做下面两件事：

### A. 先补“正常退出 / 恢复”

目标：
- 保证 `traffic_light_live_runner` 在停止后能再次启动；
- 避免 `Grp(0) is occupied` 成为新的调试摩擦点。

### B. 再把交通灯推理逻辑接到 analysis 通道

目标：
- 在当前已经稳定的 `640x640 BGR planar` analysis 通道上，接入你已有的 low-level traffic-light inference；
- 第一版先打印：
  - 检测数量
  - 类别
  - 推理耗时
- 再下一步才做叠框与 RTSP 可视化。

## 当前阶段结论

> **截至这轮，项目已经不再停留在“实时视频链路规划”阶段，而是正式进入“live skeleton 已跑通、准备接推理与补稳定性”的实施阶段。原先最关键的板端 early-init `SIGSEGV` 已经通过头文件 / ABI 对齐定位并实质性跨过。**
