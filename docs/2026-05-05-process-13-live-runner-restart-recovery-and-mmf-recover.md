# 过程 13｜traffic-light-live-runner 重启恢复路径收口与 MMF 脏状态恢复

## 本轮目标

在 `过程 12` 已经把 `traffic-light-live-runner` 从 early-init `SIGSEGV` 推进到可稳定跑起 live skeleton 之后，这一轮要回答两个更工程化的问题：

1. runner 在正常停止后能不能立即再次启动；
2. 如果调试阶段被外部硬杀，板端残留的 VPSS / MMF 脏状态怎样恢复，避免每次都重启开发板。

## 环境与对象

- 工作区：`~/repo/github/globi`
- 主程序：`board-runtime/traffic-light-live-runner`
- 目标板：Milk-V DuoS / CV181x
- 连接方式：`ssh globi`
- 板端运行文件：`/tmp/traffic_light_live_runner`
- 新增恢复脚本：`board-runtime/traffic-light-live-runner/scripts/duos_mmf_recover.sh`

## 这轮先确认了什么

### 1. 问题不在正常退出主路径，而在“未处理信号 / 硬杀”

先前看到的现象是：

```text
[SAMPLE_COMM_VPSS_Init]-28: CVI_VPSS_CreateGrp(grp:0) failed with 0xc0068004!
vpss_create_grp:3331(): Grp(0) is occupied
```

这会让人直觉以为 `SAMPLE_TDL_Destroy_MW` 本身不可靠，但这一轮实际复现实验后，结论更细：

- `SIGTERM` 下，runner 能完成线程退出和 middleware 销毁；
- 问题主要出在：
  - `SIGHUP` 之前没有被程序接管；
  - `SIGKILL` / 调试器硬杀本身不可清理，会直接留下 VPSS group 0 残留。

也就是说，之前的“重启不稳定”不是一个单一故障，而是**退出方式不同，结果不同**。

## 最小修复：把 `SIGHUP` 纳入 graceful shutdown

`main.c` 原先只处理：

- `SIGINT`
- `SIGTERM`

这一轮做的最小修复是：

1. 在 `handle_signal()` 中补 `SIGHUP` 忽略注册；
2. 启动阶段增加 `signal(SIGHUP, handle_signal);`
3. 在收到信号时显式 `fflush(stdout)`，保证退出证据不被缓冲吞掉；
4. 用 `g_exit_requested` 做一次性门闩，避免重复进入退出路径。

修复后的信号处理行为变成：

- `SIGTERM` -> graceful cleanup
- `SIGHUP` -> graceful cleanup
- `SIGKILL` -> 无法拦截，必须依赖外部恢复

## 实测验证结果

### 1. `SIGTERM` 已验证可正常清理并立即重启

板端实测日志包含：

```text
[signal] received 15, exiting...
[analysis-probe] stopped
[rtsp-output] stopped
destroy middleware
[live-skeleton] shutdown complete
```

停止后再次立即启动，runner 可以重新完成：

- VI / ISP 初始化
- VPSS group 创建
- RTSP / analysis 线程启动

说明正常退出路径已经闭环。

### 2. `SIGHUP` 修复前确实会把 VPSS 留脏

修复前用：

```bash
timeout -s HUP 5 /tmp/traffic_light_live_runner ...
```

随后立即重启，稳定复现：

```text
[SAMPLE_COMM_VPSS_Init]-28: CVI_VPSS_CreateGrp(grp:0) failed with 0xc0068004!
init vpss group failed. s32Ret: 0xffffffff !
```

同时 `/proc/cvitek/vpss` 中能看到 grp 0 仍然占用，证明问题不是猜测，而是板端状态确实没清掉。

### 3. `SIGHUP` 修复后，graceful cleanup + immediate restart 已通过

补完 `SIGHUP` 处理后，再用：

```bash
timeout -s HUP 5 /tmp/traffic_light_live_runner ...
```

这次日志已经出现：

```text
[signal] received 1, exiting...
[analysis-probe] stopped
[rtsp-output] stopped
destroy middleware
[live-skeleton] shutdown complete
```

之后立即重启，runner 可以再次正常初始化和跑流。

这说明：

> 对当前 live runner 而言，`SIGHUP` 不再是会把 VPSS 留脏的异常路径，而已经被收进正常退出流程。

### 4. `SIGKILL` 仍然会留下 MMF / VPSS 脏状态

这一点没有“代码级修复空间”，因为 `SIGKILL` 无法在进程内捕获。

实际验证过程：

1. 用 `timeout -s KILL 4 /tmp/traffic_light_live_runner ...` 制造硬杀；
2. 立即重启；
3. 稳定复现 `Grp(0) is occupied` / `0xc0068004`。

所以当前准确结论是：

- **graceful stop 已经稳定；**
- **hard kill 后的脏状态恢复必须交给外部脚本。**

## 新增板端恢复脚本

为了不再依赖重启板子，这一轮新增：

- 本地：`board-runtime/traffic-light-live-runner/scripts/duos_mmf_recover.sh`
- 板端部署路径：`/mnt/data/duos_mmf_recover.sh`

脚本做三件事：

1. 结束常见占用进程
   - `traffic_light_live_runner`
   - `traffic_light_live_runner_dbg`
   - `gdb`
   - 若存在，也一并清理常见 sample / supervisor 进程
2. 卸载并重载 MMF 相关内核模块
   - 包括 `cv181x_vpss`、`cv181x_vi`、`cv181x_sys` 等关键模块
   - 最后调用 `/mnt/system/ko/loadsystemko.sh`
3. 打印恢复前后的 `/proc/cvitek/vpss` 与 `/proc/cvitek/sys`
   - 便于直接确认 grp 占用是否被清掉

## 脚本验证结果

在故意制造 `SIGKILL` 脏状态后：

1. 直接重启 runner：失败，出现 `0xc0068004`；
2. 执行 `/mnt/data/duos_mmf_recover.sh`；
3. 再次启动 runner：成功重新跑起，并且 `timeout -s TERM 5` 正常退出。

这证明当前恢复策略已经足够支撑后续继续开发：

- 平时正常调试：直接 `SIGTERM` / Ctrl+C / `SIGHUP` 停止即可；
- 如果被 `SIGKILL` / gdb 强制打断：执行恢复脚本，不必重启整板。

## 当前项目阶段更新

相对 `过程 12`，项目状态进一步前进为：

- live skeleton 稳定运行：已完成
- graceful stop / immediate restart：已完成
- hard-kill 后 MMF 恢复手段：已完成第一版脚本化
- 真正接入 traffic-light inference：下一主任务

如果用一句话概括当前阶段：

> `traffic-light-live-runner` 已经从“能跑起来”推进到“正常停、正常起、硬杀后也有明确恢复动作”，live 视频骨架阶段基本收口，可以开始往 analysis 通道接真正的交通灯推理。

## 当前仍然成立的边界

这轮并没有解决这些问题：

1. `SIGKILL` 本身无法变成 graceful cleanup；
2. 板端 ISP 仍会打印一些已知噪声日志，例如：
   - `sensorName mismatch`
   - `AF_GetAttr / AF_SetAttr pstFocusMpiAttr is NULL`
   - 若干 `JSON_READ_ERR`

但这些日志在当前验证中**不影响 live skeleton 正常运行与退出恢复结论**。

## 下一步

下一步应该直接切主线，不要再在恢复路径上过度停留：

1. 在 `VPSS_CHN1` 的 `640x640 BGR planar` analysis 通道接入 traffic-light runtime；
2. 先做“每 N 帧推理一次 + 控制台打印结果”；
3. 稳定后再考虑把检测框叠回 RTSP 输出；
4. 把恢复脚本纳入后续演示 / 调试 SOP，减少板端被硬杀后的摩擦。
