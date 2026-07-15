# Milk-V DuoS official development handbook

Date: 2026-07-05

This is the reusable entry document for DuoS work in this workspace.  Its purpose is to keep future development anchored to Milk-V official documentation and official sample code, instead of debugging from memory or piling one-off experiments on top of each other.

## Scope

Target board:

- Milk-V DuoS / SG2000.
- Primary OS line: Buildroot SDK V2.
- Primary CPU target for this project: RISC-V 64-bit musl.
- Camera used in the current project: CAM-GC2083 on the DuoS CSI connector.
- AI runtime target: CV181x/SG2000 TPU, `.cvimodel`.

Out of scope for this handbook:

- General Linux tutorial material.
- Non-Duo Milk-V boards.
- Long historical experiment logs. Those remain in dated `docs/2026-*` files.

## Core Rules

1. Use the official V2 path for DuoS.

   DuoS work starts from `duo-buildroot-sdk-v2`, `duo-tdl-examples`, and the Milk-V Duo docs.  Older V1 SDK material is only a compatibility reference.

2. Keep firmware, application, model, and experiment artifacts separate.

   Firmware image work belongs in `duo-buildroot-sdk-v2`.  Live application code belongs in `board-runtime/traffic-light-live-runner`.  Model conversion outputs belong in `duo-tpu/workspace`.  Training data and training runs belong in `training`.

3. Prefer official camera, VPSS, RTSP, TDL, and overlay code.

   For camera live applications, start from `duo-tdl-examples/sample_vi_od/sample_vi_od.c`.  Do not hand-roll middleware lifecycle, RTSP push, VPSS setup, or OSD unless the official API cannot cover the requirement.

4. Treat data contracts as part of the design.

   If a model was compiled with `--pixel_format rgb`, the live VPSS analysis output and runtime input must remain RGB.  In this project that means `PIXEL_FORMAT_RGB_888_PLANAR` and `CVI_NN_PIXEL_RGB_PLANAR`.

5. Do board-link checks before runtime debugging.

   If USB-NCM, SSH, serial, or RTSP is down, fix that layer before changing model/runtime code.

## Official Source Map

| Source | Local path | Use it for |
|---|---|---|
| Milk-V Duo docs: https://milkv.io/zh/docs/duo | `milkv.io/docs/duo` | Board features, boot/setup, USB-NCM, SDK selection, TDL tutorials, camera docs |
| Milk-V website repo: https://github.com/milk-v/milkv.io | `milkv.io` | Local searchable copy of official docs |
| Buildroot SDK V2: https://github.com/milkv-duo/duo-buildroot-sdk-v2 | `duo-buildroot-sdk-v2` | Firmware/system image builds, V2 system libraries, `tdl_sdk`, `cvi_mpi`, rootfs/profile work |
| TDL examples: https://github.com/milkv-duo/duo-tdl-examples | `duo-tdl-examples` | Official V2 sample applications, especially camera + AI + RTSP patterns |
| Model zoo: https://github.com/milkv-duo/tdl-models | `tdl-models` | Known-good `.cvimodel` files for CV180x/CV181x |
| TPU-MLIR: https://github.com/milkv-duo/tpu-mlir | `tpu-mlir` | ONNX/MLIR/cvimodel conversion toolchain and reference docs |
| Basic Duo examples: https://github.com/milkv-duo/duo-examples | `duo-examples` | GPIO, PWM, SPI, I2C, mailbox, hello-world style examples |
| Duo files: https://github.com/milkv-duo/duo-files | `duo-files` | Released files, board-specific assets, firmware-related downloads |
| Multimedia examples | `duo-multimedia-examples` | Video player/recorder style multimedia references |

Current local official-repo snapshot:

| Repo | Branch | Head |
|---|---:|---:|
| `milkv.io` | `main` | `f98786a` |
| `duo-buildroot-sdk-v2` | `main` | `9ffb09706` |
| `duo-tdl-examples` | `master` | `6f0d2d1` |
| `tdl-models` | `milkv` | `a8a57e5` |
| `tpu-mlir` | `master` | `7fe05c0` |
| `duo-examples` | `main` | `dd39bb3` |

## Official Documentation Topics

Use these local docs first because they are searchable and versioned in the workspace:

| Topic | Local doc | Key takeaways |
|---|---|---|
| DuoS board | `milkv.io/docs/duo/getting-started/duos.md` | DuoS uses SG2000, has 512 MB memory, onboard Wi-Fi/BT, USB host, Ethernet, dual CSI, DSI, and RISC-V/ARM boot switch. |
| USB/SSH setup | `milkv.io/docs/duo/getting-started/setup.md` | Default USB networking is CDC-NCM with board IP `192.168.42.1`; SSH target is `root@192.168.42.1`. |
| Firmware build | `milkv.io/docs/duo/getting-started/buildroot-sdk.md` | DuoS should use Buildroot SDK V2; normal entry is `./build.sh lunch`. |
| TDL SDK | `milkv.io/docs/duo/application-development/tdl-sdk/tdl-sdk-introduction.md` | TDL wraps common AI algorithm pre/postprocessing and provides a unified API; V1 and V2 firmware use different sample paths. |
| YOLO TDL docs | `milkv.io/docs/duo/application-development/tdl-sdk/tdl-sdk-yolov5.md`, `tdl-sdk-yolov8.md`, `tdl-sdk-yolo11.md`, `tdl-sdk-yolo12.md` | Official model conversion flow is ONNX -> `model_transform.py` -> MLIR -> calibration -> `model_deploy.py` -> `.cvimodel`; examples use RGB pixel format. |
| CAM-GC2083 | `milkv.io/docs/duo/camera/gc2083.md` | GC2083 is the official 2MP CSI camera path; sample output uses RTSP at `rtsp://192.168.42.1/h264`. |
| TPU intro | `milkv.io/docs/duo/application-development/tpu/tpu-introduction.md` | For practical TPU application development, prefer TDL SDK unless a lower-level runtime path is needed. |

## Workspace Map

Use this map before editing anything:

| Path | Role | Edit policy |
|---|---|---|
| `DUOS_BASELINE.md` | Short project entrypoint | Keep it short; point to deeper docs |
| `docs/duos-official-development-handbook.md` | This reusable handbook | Update when source maps or project rules change |
| `docs/2026-07-05-duos-official-baseline-audit.md` | Current audit of our live-runner baseline | Update when traffic live-runner architecture changes |
| `tools/check_duos_official_baseline.sh` | Host-side consistency check | Keep fast and conservative |
| `tools/diagnose_duos_link.ps1` | Windows board-link diagnosis | Use before board deployment; do not use when board is unplugged |
| `duo-buildroot-sdk-v2` | Firmware/system SDK | Use for image/rootfs/system changes only |
| `duo-tdl-examples` | Official V2 application examples | Treat as reference; avoid local modifications except deliberate experiments |
| `board-runtime/minimal-cviruntime-runner` | Static-image model/runtime validation | Use for isolated model and postprocess debugging |
| `board-runtime/traffic-light-live-runner` | Custom live traffic-light app | Keep aligned with `sample_vi_od` structure |
| `duo-tpu/workspace` | TPU conversion outputs | Do not mix with source code |
| `training/traffic_light_yolo` | Traffic-light dataset | Training/evaluation only |

## Firmware Workflow

Use this when the OS image, rootfs, services, or system libraries need to change.

1. Start from official SDK V2.

   ```bash
   cd ~/repo/github/globi/duo-buildroot-sdk-v2
   ./build.sh lunch
   ```

2. Select the DuoS RISC-V target unless there is a specific reason to build ARM.

3. Keep image output and flashing notes in dated docs, not in application READMEs.

4. Do not debug live application behavior by changing firmware first unless the failure is clearly system-level.

## Official TDL Application Workflow

Use this when building a camera + AI + RTSP app.

1. Compile and run the official example before custom code.

   ```bash
   cd ~/repo/github/globi/duo-tdl-examples
   source envsetup.sh
   cd sample_vi_od
   make
   ```

2. Read `sample_vi_od.c` as the live-app skeleton:

   - middleware config from `/mnt/data/sensor_cfg.ini`
   - VI -> VPSS group
   - one channel for VENC/RTSP
   - one channel for AI
   - shared detection metadata guarded by a mutex
   - RTSP overlay through `CVI_TDL_Service_ObjectDrawRect`
   - model lifecycle through `CVI_TDL_CreateHandle2`, `CVI_TDL_OpenModel`, and cleanup functions

3. For custom models, change the smallest possible piece:

   - Keep official middleware, VPSS, RTSP, service handle, and thread lifecycle.
   - Replace only the official TDL inference call when the official TDL API does not support the custom model.
   - Convert custom detection output into official `cvtdl_object_t` metadata for drawing.

## Custom Model Conversion Workflow

Use this when training/exporting a new YOLO-family traffic-light model.

1. Keep training outputs in `training`.

   Current dataset root:

   ```bash
   training/traffic_light_yolo/dataset.yaml
   ```

2. Convert under `duo-tpu/workspace/<model-name>`.

3. Follow the official TPU-MLIR shape:

   ```text
   ONNX -> model_transform.py -> MLIR
   MLIR + calibration_table -> model_deploy.py -> cvimodel
   ```

4. Keep the input contract explicit.

   For our current live runner and official YOLO examples:

   ```text
   pixel format: rgb
   live VPSS:    PIXEL_FORMAT_RGB_888_PLANAR
   cviruntime:   CVI_NN_PIXEL_RGB_PLANAR
   ```

5. Treat live misclassification as model/data/eval only after the board-side RGB path is proven.

## Current Traffic-Light Project Baseline

The current live project is here:

```bash
board-runtime/traffic-light-live-runner
```

The intended architecture is:

```text
GC2083 camera
  -> VI
  -> VPSS group 0
     -> CHN0 NV21 1280x720 -> H.264 RTSP
     -> CHN1 RGB planar 640x640 -> CVI_NN_SetTensorWithAlignedFrames -> TPU inference
                                        -> YOLO decode/NMS
                                        -> cvtdl_object_t
                                        -> CVI_TDL_Service_ObjectDrawRect
```

The live runner must preserve these invariants:

- No `PIXEL_FORMAT_BGR_888_PLANAR`.
- No `CVI_NN_PIXEL_BGR_PLANAR`.
- No CPU BGR->RGB conversion in the live path.
- No hand-written NV21/YUV box drawing when the TDL service can draw the box.
- No leftover region-overlay cleanup such as blind `CVI_RGN_Destroy(0/1)` from earlier experiments.
- Local build must link official TDL/OpenCV/IVE dependencies.

Check it with:

```bash
cd ~/repo/github/globi/board-runtime/traffic-light-live-runner
./scripts/check_live_runner_baseline.sh
```

Deploy when the board is connected:

```bash
cd ~/repo/github/globi/board-runtime/traffic-light-live-runner
./scripts/deploy_live_runner.sh --run
```

## Board-Link Workflow

Use this only when the board is connected.

1. Confirm Windows sees the USB network device or another intended network path.

2. Check the link:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools/diagnose_duos_link.ps1
   ```

3. Only after the link is healthy, deploy application code.

4. Use VLC or another RTSP client:

   ```text
   rtsp://192.168.42.1/h264
   ```

If the board is unplugged, do not spend time interpreting SSH/serial failures.

## Verification Gates

Run the full local baseline before and after meaningful changes:

```bash
cd ~/repo/github/globi
./tools/check_duos_official_baseline.sh
```

This currently checks:

- required official/local repositories exist
- key official docs are present locally
- `duo-tdl-examples/sample_vi_od` still contains the expected official overlay path
- official `sample_vi_od` can be rebuilt for DuoS/RISCV64
- traffic live runner preserves RGB input, official overlay, and successful build

## Decision Guide

Use official TDL end-to-end when:

- The model type is directly supported by `CVI_TDL_*`.
- The official model zoo contains a close-enough model.
- The project is mainly camera/RTSP/demo integration.

Use low-level cviruntime only when:

- The model is custom and not directly covered by the official TDL model wrapper.
- You need to control YOLO decode/NMS/class mapping yourself.
- You can still reuse official middleware, RTSP, VPSS, and overlay paths.

Use firmware work only when:

- You need to change rootfs contents, boot behavior, drivers, system services, or packaged libraries.
- Application deployment alone cannot solve the problem.

## Known Local Artifacts Worth Keeping

- `duo-tpu/workspace/traffic_light_yolov8n_named/work/traffic_light_yolov8n_named_cv181x_int8_sym.cvimodel`
- `duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_cv181x_int8_sym.cvimodel`
- `duo-tpu/workspace/traffic_light_yolov5s/work/traffic_light_yolov5s_split_cv181x_int8_sym.cvimodel`
- `board-runtime/minimal-cviruntime-runner`
- `board-runtime/traffic-light-live-runner`
- `training/traffic_light_yolo/dataset.yaml`

## Anti-Drift Checklist

Before making code changes, answer these:

- Which official doc or official sample is the closest source of truth?
- Am I editing firmware, application code, model conversion, or training data?
- Is the board link healthy, or am I debugging a disconnected device?
- Does this change preserve the RGB model input contract?
- Can `tools/check_duos_official_baseline.sh` prove the local baseline still holds?
