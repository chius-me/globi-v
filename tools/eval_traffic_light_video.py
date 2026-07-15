#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


CLASS_NAMES = ["red_light", "yellow_light", "green_light", "off"]
ANCHORS = [
    [(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)],
    [(30.0, 61.0), (62.0, 45.0), (59.0, 119.0)],
    [(116.0, 90.0), (156.0, 198.0), (373.0, 326.0)],
]
STRIDES = [8, 16, 32]


@dataclass
class PrepInfo:
    orig_w: int
    orig_h: int
    target_w: int
    target_h: int
    resized_w: int
    resized_h: int
    paste_x: int
    paste_y: int


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox_bgr(image: np.ndarray, size: int = 640, pad_value: int = 0) -> Tuple[np.ndarray, PrepInfo]:
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)
    resized_w = max(1, int(round(w * ratio)))
    resized_h = max(1, int(round(h * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    paste_x = (size - resized_w) // 2
    paste_y = (size - resized_h) // 2
    canvas[paste_y:paste_y + resized_h, paste_x:paste_x + resized_w] = resized
    return canvas, PrepInfo(w, h, size, size, resized_w, resized_h, paste_x, paste_y)


def to_model_input(image_bgr: np.ndarray) -> Tuple[np.ndarray, PrepInfo]:
    prepared, prep = letterbox_bgr(image_bgr, 640, 0)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return nchw[None, ...], prep


def restore_box(box: Sequence[float], prep: PrepInfo) -> List[float]:
    x1, y1, x2, y2 = box
    sx = prep.orig_w / prep.resized_w
    sy = prep.orig_h / prep.resized_h
    x1 = (x1 - prep.paste_x) * sx
    x2 = (x2 - prep.paste_x) * sx
    y1 = (y1 - prep.paste_y) * sy
    y2 = (y2 - prep.paste_y) * sy
    return [
        float(np.clip(x1, 0, prep.orig_w - 1)),
        float(np.clip(y1, 0, prep.orig_h - 1)),
        float(np.clip(x2, 0, prep.orig_w - 1)),
        float(np.clip(y2, 0, prep.orig_h - 1)),
    ]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(a[0], b[:, 0])
    inter_y1 = np.maximum(a[1], b[:, 1])
    inter_x2 = np.minimum(a[2], b[:, 2])
    inter_y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    denom = area_a + area_b - inter
    return np.where(denom > 0.0, inter / denom, 0.0)


def nms(dets: List[Dict], iou_thres: float, max_det: int) -> List[Dict]:
    final: List[Dict] = []
    for class_id in range(len(CLASS_NAMES)):
        cls = [d for d in dets if d["class_id"] == class_id]
        cls.sort(key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict] = []
        while cls:
            best = cls.pop(0)
            kept.append(best)
            if not cls:
                break
            boxes = np.array([d["xyxy_640"] for d in cls], dtype=np.float32)
            ious = box_iou(np.array(best["xyxy_640"], dtype=np.float32), boxes)
            cls = [d for d, iou in zip(cls, ious) if iou <= iou_thres]
        final.extend(kept)
    final.sort(key=lambda d: d["confidence"], reverse=True)
    return final[:max_det]


def finalize_detections(dets: List[Dict], prep: PrepInfo, iou: float, max_det: int) -> List[Dict]:
    out = nms(dets, iou, max_det)
    for det in out:
        det["xyxy"] = restore_box(det["xyxy_640"], prep)
        det["class_name"] = CLASS_NAMES[det["class_id"]]
        del det["xyxy_640"]
    return out


def decode_onnx_split(outputs: Sequence[np.ndarray], prep: PrepInfo, conf: float, iou: float, max_det: int) -> List[Dict]:
    dets: List[Dict] = []
    for scale_idx in range(3):
        bbox = outputs[scale_idx * 3 + 0]
        obj = outputs[scale_idx * 3 + 1]
        cls = outputs[scale_idx * 3 + 2]
        h, w = bbox.shape[1], bbox.shape[2]
        stride = STRIDES[scale_idx]
        anchors = ANCHORS[scale_idx]
        for a in range(3):
            obj_s = sigmoid(obj[a, :, :, 0])
            cls_s = sigmoid(cls[a])
            scores = obj_s[:, :, None] * cls_s
            best_cls = np.argmax(scores, axis=2)
            best_conf = np.max(scores, axis=2)
            ys, xs = np.where(best_conf >= conf)
            for y, x in zip(ys.tolist(), xs.tolist()):
                c = int(best_cls[y, x])
                raw = bbox[a, y, x]
                cx = (float(sigmoid(raw[0])) * 2.0 - 0.5 + x) * stride
                cy = (float(sigmoid(raw[1])) * 2.0 - 0.5 + y) * stride
                bw = (float(sigmoid(raw[2])) * 2.0) ** 2 * anchors[a][0]
                bh = (float(sigmoid(raw[3])) * 2.0) ** 2 * anchors[a][1]
                dets.append({
                    "class_id": c,
                    "confidence": float(best_conf[y, x]),
                    "xyxy_640": [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
                })
    return finalize_detections(dets, prep, iou, max_det)


def decode_pt_pred(pred: np.ndarray, prep: PrepInfo, conf: float, iou: float, max_det: int) -> List[Dict]:
    dets: List[Dict] = []
    pred = pred[0]
    obj = pred[:, 4]
    cls = pred[:, 5:5 + len(CLASS_NAMES)]
    scores = obj[:, None] * cls
    best_cls = np.argmax(scores, axis=1)
    best_conf = np.max(scores, axis=1)
    keep = np.where(best_conf >= conf)[0]
    for idx in keep.tolist():
        cx, cy, w, h = pred[idx, :4].tolist()
        dets.append({
            "class_id": int(best_cls[idx]),
            "confidence": float(best_conf[idx]),
            "xyxy_640": [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
        })
    return finalize_detections(dets, prep, iou, max_det)


def draw(image: np.ndarray, detections: Sequence[Dict]) -> np.ndarray:
    colors = [(0, 0, 255), (0, 215, 255), (0, 200, 0), (160, 160, 160)]
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = colors[det["class_id"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{det["class_name"]}:{det["confidence"]:.2f}'
        cv2.putText(out, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def load_pt_model(weights: str, yolov5_dir: str):
    import torch

    pathlib.WindowsPath = pathlib.PosixPath
    sys.path.insert(0, yolov5_dir)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval()
    return model


def iter_sampled_frames(video_path: pathlib.Path, every_sec: float, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_sec)))
    wanted = list(range(0, total if total > 0 else step * max_frames, step))[:max_frames]
    for frame_idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            yield frame_idx, frame_idx / fps, frame
    cap.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--pt")
    parser.add_argument("--onnx")
    parser.add_argument("--yolov5-dir", default="duo-tpu/workspace/traffic_light_teammate_yolov5s/yolov5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every-sec", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--annotate-video", action="store_true",
                        help="Write full annotated videos using ONNX results.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    frames_dir = out_dir / "frames"
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pt_model = load_pt_model(args.pt, args.yolov5_dir) if args.pt else None
    torch = __import__("torch") if pt_model is not None else None

    ort_session = None
    if args.onnx:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
    else:
        input_name = ""

    rows = []
    details = []
    for video in [pathlib.Path(v) for v in args.videos]:
        for frame_idx, ts, frame in iter_sampled_frames(video, args.every_sec, args.max_frames_per_video):
            stem = f"{video.stem}_f{frame_idx:06d}"
            frame_path = frames_dir / f"{stem}.jpg"
            cv2.imwrite(str(frame_path), frame)
            inp, prep = to_model_input(frame)

            model_results = {}
            if pt_model is not None:
                with torch.no_grad():
                    pred = pt_model(torch.from_numpy(inp))[0].detach().cpu().numpy()
                model_results["pt"] = decode_pt_pred(pred, prep, args.conf, args.iou, args.max_det)
            if ort_session is not None:
                outputs = ort_session.run(None, {input_name: inp})
                model_results["onnx"] = decode_onnx_split(outputs, prep, args.conf, args.iou, args.max_det)

            for model_name, dets in model_results.items():
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
                    "count": len(dets),
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

    if args.annotate_video:
        if ort_session is None:
            raise RuntimeError("--annotate-video currently requires --onnx")
        video_out_dir = out_dir / "videos"
        video_out_dir.mkdir(parents=True, exist_ok=True)
        for video in [pathlib.Path(v) for v in args.videos]:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video: {video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            tmp_path = video_out_dir / f"{video.stem}_annotated_raw.mp4"
            out_path = video_out_dir / f"{video.stem}_annotated.mp4"
            writer = cv2.VideoWriter(
                str(tmp_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                inp, prep = to_model_input(frame)
                outputs = ort_session.run(None, {input_name: inp})
                dets = decode_onnx_split(outputs, prep, args.conf, args.iou, args.max_det)
                writer.write(draw(frame, dets))
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"annotated {video.name}: {frame_idx}/{total or '?'} frames")
            writer.release()
            cap.release()

            import subprocess

            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(tmp_path),
                "-i", str(video),
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

    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'details.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
