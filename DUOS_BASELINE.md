# DuoS baseline

This repository uses the official Milk-V DuoS V2 path as the development baseline. The upstream SDKs and model files listed below are external prerequisites and are not stored in this repository.

Start here:

```bash
tools/check_duos_official_baseline.sh
```

That command verifies:

- local official repository map: `milkv.io`, `duo-buildroot-sdk-v2`, `duo-tdl-examples`, `tdl-models`, `tpu-mlir`
- local Milk-V documentation snapshots for USB-NCM/SSH and Buildroot SDK V2
- official `duo-tdl-examples/sample_vi_od` build for DuoS/RISCV64
- traffic live runner baseline: RGB planar input, cviruntime feed, official TDL service overlay, and build

Primary notes:

- Reusable handbook: `docs/duos-official-development-handbook.md`
- Full audit: `docs/2026-07-05-duos-official-baseline-audit.md`
- Live runner: `board-runtime/traffic-light-live-runner/`
- Live runner local check: `board-runtime/traffic-light-live-runner/scripts/check_live_runner_baseline.sh`
- Live runner deploy/smoke entry: `board-runtime/traffic-light-live-runner/scripts/deploy_live_runner.sh`
- Windows host link diagnosis: `tools/diagnose_duos_link.ps1`

Current rule of thumb:

- Firmware/system image work belongs in `duo-buildroot-sdk-v2`.
- Camera/RTSP/TDL live app structure should follow `duo-tdl-examples/sample_vi_od`.
- Custom traffic-light logic is limited to cviruntime model loading and YOLO decode/NMS in `board-runtime/traffic-light-live-runner`.
- Board access should use the `milkv-duo` SSH alias (`root@192.168.42.1`) unless explicitly overridden.
- Before debugging runtime code, run `powershell -ExecutionPolicy Bypass -File tools/diagnose_duos_link.ps1` and fix USB/SSH/serial failures first.
