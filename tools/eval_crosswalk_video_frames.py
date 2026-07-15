#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import sys

import cv2

from eval_traffic_light_images import (
    CLASS_NAMES,
    decode_onnx_split,
    decode_pt,
    draw,
    load_pt_model,
    to_input,
)


def iter_sampled_frames(video_path: pathlib.Path, every_sec: float, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_sec)))
    frame_indices = list(range(0, total if total > 0 else step * max_frames, step))[:max_frames]
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            yield frame_idx, frame_idx / fps, frame
    cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--yolov5-dir", default="duo-tpu/workspace/traffic_light_crosswalk_yolov5s/yolov5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every-sec", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    out_dir = pathlib.Path(args.out)
    frames_dir = out_dir / "frames"
    vis_dir = out_dir / "vis"
    frames_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pt_model = load_pt_model(pathlib.Path(args.pt), pathlib.Path(args.yolov5_dir))
    ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    rows = []
    details = []
    for video in [pathlib.Path(v) for v in args.videos]:
        for frame_idx, ts, frame in iter_sampled_frames(video, args.every_sec, args.max_frames_per_video):
            stem = f"{video.stem}_f{frame_idx:06d}"
            frame_path = frames_dir / f"{stem}.jpg"
            cv2.imwrite(str(frame_path), frame)
            inp, prep = to_input(frame)

            with torch.no_grad():
                pt_pred = pt_model(torch.from_numpy(inp))[0].detach().cpu().numpy()
            onnx_outputs = ort_session.run(None, {input_name: inp})
            results = {
                "pt": decode_pt(pt_pred, prep, args.conf, args.iou, args.max_det),
                "onnx": decode_onnx_split(onnx_outputs, prep, args.conf, args.iou, args.max_det),
            }

            for model_name, dets in results.items():
                vis_path = vis_dir / f"{stem}_{model_name}.jpg"
                cv2.imwrite(str(vis_path), draw(frame, dets))
                counts = {name: 0 for name in CLASS_NAMES}
                for det in dets:
                    counts[det["class_name"]] += 1
                rows.append({
                    "video": video.name,
                    "frame": frame_idx,
                    "time_sec": round(ts, 3),
                    "model": model_name,
                    "detections": len(dets),
                    **counts,
                    "top": "; ".join(f'{d["class_name"]}:{d["confidence"]:.3f}' for d in dets[:5]),
                    "frame_path": str(frame_path),
                    "vis_path": str(vis_path),
                })
                details.append({
                    "video": video.name,
                    "frame": frame_idx,
                    "time_sec": ts,
                    "model": model_name,
                    "frame_path": str(frame_path),
                    "vis_path": str(vis_path),
                    "detections": dets,
                })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["video"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"videos={len(args.videos)} rows={len(rows)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
