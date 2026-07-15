# DuoS official baseline audit

Date: 2026-07-05

This note is the current development baseline for the Milk-V DuoS traffic-light work.  It is intentionally conservative: use official Milk-V/Sophgo code for board bring-up, camera, RTSP, VPSS, and OSD whenever possible; keep custom code only where the official samples do not cover our custom traffic-light `cvimodel`.

## Board and OS baseline

- Board: Milk-V DuoS / SG2000.
- Camera: GC2083, 1920x1080 30fps RGB Bayer sensor.
- OS/SDK line: Buildroot SDK V2, not V1.
- Board target we have been using: `milkv-duos-musl-riscv64-emmc`.
- USB network expectation: CDC-NCM + DHCP, board at `192.168.42.1`, SSH as `root`.

Official basis:

- `milkv.io/docs/duo/getting-started/setup.md`: default USB networking is CDC-NCM; SSH target is `root@192.168.42.1`.
- `milkv.io/docs/duo/getting-started/buildroot-sdk.md`: SDK V2 is recommended for Duo256M/DuoS and contains Buildroot 2025.02, Linux 5.10.4, `cvi_mpi`, and `tdl_sdk`.
- `duo-buildroot-sdk-v2` local clone: official firmware/build source. Current working tree has local image-size/profile edits; treat that repo as firmware work only.

## Application baseline

For camera + AI + RTSP live applications, use the V2 TDL examples as the first reference:

- `duo-tdl-examples/README-zh.md`
- `duo-tdl-examples/sample_vi_fd/sample_vi_fd.c`
- `duo-tdl-examples/sample_vi_od/sample_vi_od.c`
- `duo-buildroot-sdk-v2/tdl_sdk/sample_video/*`

The official `sample_vi_od` shape is the right live-app architecture:

- middleware config from `/mnt/data/sensor_cfg.ini`
- VI -> VPSS group 0
- channel 0 -> VENC/RTSP
- channel 1 -> AI thread
- shared detection metadata protected by a mutex
- RTSP thread draws boxes with `CVI_TDL_Service_ObjectDrawRect`
- inference thread updates the shared result and releases VPSS frames

For overlays, the official path is `CVI_TDL_Service_ObjectDrawRect` / related TDL service functions.  Hand-written NV21 Y/UV drawing should be treated as prototype-only unless the service API cannot represent our detections.

## Model conversion baseline

The model-input contract must remain RGB:

- Official YOLOv5 TDL documentation uses `model_transform.py --pixel_format rgb`, `--scale 0.0039216,0.0039216,0.0039216`, and `--keep_aspect_ratio`.
- Official TPU YOLOv5 documentation also states RGB model input.
- Therefore live VPSS analysis output and runtime input must be RGB planar:
  - VPSS analysis channel: `PIXEL_FORMAT_RGB_888_PLANAR`
  - cviruntime feed: `CVI_NN_PIXEL_RGB_PLANAR`

This means yellow-as-green must not be fixed by swapping BGR/RGB again.  If yellow is still misclassified after the RGB path is proven on-board, the likely causes are model/data/calibration/class balance or postprocessing thresholds, not a declared BGR input path.

## Local repository map

Official or near-official local clones:

- `milkv.io`: local copy of Milk-V docs.
- `duo-buildroot-sdk-v2`: official V2 firmware SDK.
- `duo-tdl-examples`: official V2 app examples; use this as the primary camera/TDL sample source.
- `cvitek-tdl-sdk-sg200x`: V1/older-style SG200x SDK samples; useful for `sample/cvi_yolo` and export references, not the first choice for V2 live app structure.
- `cvitek-tdl-sdk-cv180x`: Duo 64M/CV180x reference only.
- `tdl-models`: official model files.
- `tpu-mlir`: model conversion toolchain.

Project-specific local code:

- `board-runtime/minimal-cviruntime-runner`: static-image cviruntime validation tool.  Keep as the reproducible board-side model/debug runner.
- `board-runtime/traffic-light-live-runner`: current custom live runner.  This should be treated as an integration branch, not an official-style clean baseline yet.
- `duo-tpu/workspace/traffic_light_*`: training/export/conversion artifacts.
- `training/traffic_light_yolo`: dataset/model training workspace.

## Current traffic-light live-runner status

What is worth keeping:

- Direct cviruntime loading for our custom traffic-light `cvimodel`.
- YOLO decode/NMS logic adapted from `minimal-cviruntime-runner`.
- Two-thread live shape: RTSP thread + analysis thread.
- RGB VPSS analysis channel and `CVI_NN_SetTensorWithAlignedFrames(..., CVI_NN_PIXEL_RGB_PLANAR)`.
- RTSP overlay now maps `TLLR_Detection[]` to `cvtdl_object_t` and uses the official `CVI_TDL_Service_ObjectDrawRect` path.
- `live_inference.cpp` now keeps a single live input path: RGB planar VPSS frame addresses passed directly to cviruntime.

What should be cleaned up before further feature work:

- `src/live_inference.cpp` and `src/live_inference.h` are untracked in the root worktree; core code should not remain untracked.
- Board-side deployment still needs verification after USB/SSH access returns.
- The older custom NV21 plane-write overlay has been removed from source, but still needs board-side verification after USB/SSH access returns.
- `CVI_RGN_Destroy(0/1)` cleanup from the earlier green-square overlay experiment has been removed from source, but still needs board-side verification after USB/SSH access returns.
- The README had stale BGR wording; it has been corrected to RGB.

## Recommended next implementation direction

1. Stabilize board access first.
   - Windows should show `UsbNcm Host Device` or equivalent CDC-NCM network adapter.
   - `ping 192.168.42.1` and `ssh root@192.168.42.1` must work before any board-side verification.

2. Make an official-style clean live runner.
   - Start from `duo-tdl-examples/sample_vi_od/sample_vi_od.c` structure.
   - Keep official middleware setup, thread lifecycle, result mutex, and service overlay style.
   - Replace only the official TDL inference call with our `tllr_inference_run_frame`.
   - Convert `TLLR_Detection[]` into `cvtdl_object_t` and draw with `CVI_TDL_Service_ObjectDrawRect`. This part is now implemented in the current live runner source.
   - Mapping is direct: `TLLR_Detection.{x1,y1,x2,y2,confidence,class_id}` -> `cvtdl_object_info_t.{bbox.x1,bbox.y1,bbox.x2,bbox.y2,bbox.score,classes}`, plus `name`.

3. Keep RGB zero-copy input as the only live path.
   - VPSS CHN1: `PIXEL_FORMAT_RGB_888_PLANAR`.
   - Runtime: `CVI_NN_SetTensorWithAlignedFrames(..., CVI_NN_PIXEL_RGB_PLANAR)`.
   - CPU color conversion and yellow color-refine code have been removed from the live runner source.

4. Treat yellow-light confusion as a model/eval issue after board RGB verification.
   - Capture a few live frames from the exact camera stream.
   - Run the static runner against those frames.
   - Compare red/green/yellow class logits and boxes.
   - Then decide whether to adjust thresholds, dataset balance, calibration set, or retrain.

5. Keep firmware and app work separate.
   - Firmware image changes belong in `duo-buildroot-sdk-v2`.
   - Live app changes belong in `board-runtime/traffic-light-live-runner`.
   - Model conversion records belong in `docs/` plus `duo-tpu/workspace`.

## Immediate verification gates

- Full local DuoS baseline:
  - `tools/check_duos_official_baseline.sh`.
  - This checks the local official repo map, local Milk-V documentation snapshots, official `sample_vi_od` build, and the traffic live runner baseline.
- Source scan:
  - no `PIXEL_FORMAT_BGR_888_PLANAR` or `CVI_NN_PIXEL_BGR_PLANAR` in the live runner path.
- Host link diagnosis:
  - `powershell -ExecutionPolicy Bypass -File tools/diagnose_duos_link.ps1`.
  - This checks Windows USB network/PnP state, ping/TCP to `192.168.42.1`, SSH via `milkv-duo`, and serial-port enumeration.
- Build:
  - `cd board-runtime/traffic-light-live-runner && ./build.sh`.
- Local baseline check:
  - `cd board-runtime/traffic-light-live-runner && ./scripts/check_live_runner_baseline.sh`.
  - This checks RGB VPSS/runtime input, official `CVI_TDL_Service_ObjectDrawRect` overlay, absence of old BGR/NV21/RGN/debug paths, and a successful build.
- Official sample build:
  - `duo-tdl-examples/sample_vi_od` has been rebuilt locally with the DuoS/RISCV64 host toolchain and official Makefile.
- Board deploy:
  - `cd board-runtime/traffic-light-live-runner && ./scripts/deploy_live_runner.sh`.
- Board run:
  - `ssh milkv-duo '/root/traffic-light-live/run.sh'`.
- Video:
  - VLC opens `rtsp://192.168.42.1/h264`.
  - boxes are drawn by official service path, not by custom NV21 writes.
