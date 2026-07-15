#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys

import cv2

from eval_traffic_light_images import (
    decode_onnx_split,
    decode_pt,
    draw,
    load_pt_model,
    to_input,
)


def annotate_with_onnx(video: pathlib.Path, out_path: pathlib.Path, session, input_name: str,
                       conf: float, iou: float, max_det: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tmp_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        inp, prep = to_input(frame)
        outputs = session.run(None, {input_name: inp})
        detections = decode_onnx_split(outputs, prep, conf, iou, max_det)
        writer.write(draw(frame, detections))
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"onnx {video.name}: {frame_idx}/{total or '?'}")
    writer.release()
    cap.release()
    transcode(tmp_path, video, out_path)


def annotate_with_pt(video: pathlib.Path, out_path: pathlib.Path, model, torch,
                     conf: float, iou: float, max_det: int) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tmp_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        inp, prep = to_input(frame)
        with torch.no_grad():
            pred = model(torch.from_numpy(inp))[0].detach().cpu().numpy()
        detections = decode_pt(pred, prep, conf, iou, max_det)
        writer.write(draw(frame, detections))
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"pt {video.name}: {frame_idx}/{total or '?'}")
    writer.release()
    cap.release()
    transcode(tmp_path, video, out_path)


def transcode(tmp_path: pathlib.Path, source_video: pathlib.Path, out_path: pathlib.Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_path),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ], check=True)
    tmp_path.unlink(missing_ok=True)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--yolov5-dir", default="duo-tpu/workspace/traffic_light_crosswalk_yolov5s/yolov5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["onnx", "pt", "both"], default="both")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [pathlib.Path(v) for v in args.videos]

    if args.model in {"onnx", "both"}:
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        for video in videos:
            annotate_with_onnx(
                video,
                out_dir / f"{video.stem}_onnx_conf{int(args.conf * 100):03d}.mp4",
                session,
                input_name,
                args.conf,
                args.iou,
                args.max_det,
            )

    if args.model in {"pt", "both"}:
        import torch
        model = load_pt_model(pathlib.Path(args.pt), pathlib.Path(args.yolov5_dir))
        for video in videos:
            annotate_with_pt(
                video,
                out_dir / f"{video.stem}_pt_conf{int(args.conf * 100):03d}.mp4",
                model,
                torch,
                args.conf,
                args.iou,
                args.max_det,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
