#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


CLASS_NAMES = ["red", "yellow", "green", "off", "crosswalk", "guide_arrows"]
CLASS_COLORS = [
    (0, 0, 255),
    (0, 215, 255),
    (0, 200, 0),
    (160, 160, 160),
    (255, 120, 0),
    (255, 0, 180),
]
ANCHORS = [
    [(6.12890625, 15.3203125), (10.734375, 11.296875), (9.7578125, 24.234375)],
    [(13.78125, 32.46875), (104.375, 15.9921875), (31.734375, 78.5625)],
    [(337.25, 28.703125), (508.5, 69.0625), (574.5, 107.5625)],
]
STRIDES = [8, 16, 32]


@dataclass
class PrepInfo:
    orig_w: int
    orig_h: int
    resized_w: int
    resized_h: int
    paste_x: int
    paste_y: int


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox_bgr(image: np.ndarray, size: int = 640) -> Tuple[np.ndarray, PrepInfo]:
    h, w = image.shape[:2]
    ratio = min(size / w, size / h)
    resized_w = max(1, int(round(w * ratio)))
    resized_h = max(1, int(round(h * ratio)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    paste_x = (size - resized_w) // 2
    paste_y = (size - resized_h) // 2
    canvas[paste_y:paste_y + resized_h, paste_x:paste_x + resized_w] = resized
    return canvas, PrepInfo(w, h, resized_w, resized_h, paste_x, paste_y)


def to_input(image_bgr: np.ndarray) -> Tuple[np.ndarray, PrepInfo]:
    prepared, prep = letterbox_bgr(image_bgr, 640)
    rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return nchw[None, ...], prep


def restore_box(box: Sequence[float], prep: PrepInfo) -> List[float]:
    x1, y1, x2, y2 = box
    sx = prep.orig_w / prep.resized_w
    sy = prep.orig_h / prep.resized_h
    return [
        float(np.clip((x1 - prep.paste_x) * sx, 0, prep.orig_w - 1)),
        float(np.clip((y1 - prep.paste_y) * sy, 0, prep.orig_h - 1)),
        float(np.clip((x2 - prep.paste_x) * sx, 0, prep.orig_w - 1)),
        float(np.clip((y2 - prep.paste_y) * sy, 0, prep.orig_h - 1)),
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
        candidates = [d for d in dets if d["class_id"] == class_id]
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        while candidates:
            best = candidates.pop(0)
            final.append(best)
            if not candidates:
                break
            boxes = np.array([d["xyxy_640"] for d in candidates], dtype=np.float32)
            ious = box_iou(np.array(best["xyxy_640"], dtype=np.float32), boxes)
            candidates = [d for d, iou in zip(candidates, ious) if iou <= iou_thres]
    final.sort(key=lambda d: d["confidence"], reverse=True)
    return final[:max_det]


def finalize(dets: List[Dict], prep: PrepInfo, iou: float, max_det: int) -> List[Dict]:
    out = nms(dets, iou, max_det)
    for det in out:
        det["class_name"] = CLASS_NAMES[det["class_id"]]
        det["xyxy"] = restore_box(det.pop("xyxy_640"), prep)
    return out


def decode_pt(pred: np.ndarray, prep: PrepInfo, conf: float, iou: float, max_det: int) -> List[Dict]:
    dets: List[Dict] = []
    pred = pred[0]
    obj = pred[:, 4]
    cls = pred[:, 5:5 + len(CLASS_NAMES)]
    scores = obj[:, None] * cls
    best_cls = np.argmax(scores, axis=1)
    best_conf = np.max(scores, axis=1)
    for idx in np.where(best_conf >= conf)[0].tolist():
        cx, cy, w, h = pred[idx, :4].tolist()
        dets.append({
            "class_id": int(best_cls[idx]),
            "confidence": float(best_conf[idx]),
            "xyxy_640": [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
        })
    return finalize(dets, prep, iou, max_det)


def decode_onnx_split(outputs: Sequence[np.ndarray], prep: PrepInfo, conf: float,
                      iou: float, max_det: int) -> List[Dict]:
    dets: List[Dict] = []
    for scale_idx in range(3):
        bbox = outputs[scale_idx * 3 + 0]
        obj = outputs[scale_idx * 3 + 1]
        cls = outputs[scale_idx * 3 + 2]
        h, w = bbox.shape[1], bbox.shape[2]
        stride = STRIDES[scale_idx]
        anchors = ANCHORS[scale_idx]
        for anchor_idx in range(3):
            obj_s = sigmoid(obj[anchor_idx, :, :, 0])
            cls_s = sigmoid(cls[anchor_idx])
            scores = obj_s[:, :, None] * cls_s
            best_cls = np.argmax(scores, axis=2)
            best_conf = np.max(scores, axis=2)
            ys, xs = np.where(best_conf >= conf)
            for y, x in zip(ys.tolist(), xs.tolist()):
                raw = bbox[anchor_idx, y, x]
                cx = (float(sigmoid(raw[0])) * 2.0 - 0.5 + x) * stride
                cy = (float(sigmoid(raw[1])) * 2.0 - 0.5 + y) * stride
                bw = (float(sigmoid(raw[2])) * 2.0) ** 2 * anchors[anchor_idx][0]
                bh = (float(sigmoid(raw[3])) * 2.0) ** 2 * anchors[anchor_idx][1]
                dets.append({
                    "class_id": int(best_cls[y, x]),
                    "confidence": float(best_conf[y, x]),
                    "xyxy_640": [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
                })
    return finalize(dets, prep, iou, max_det)


def draw(image: np.ndarray, detections: Sequence[Dict]) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = CLASS_COLORS[det["class_id"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{det["class_name"]}:{det["confidence"]:.2f}'
        cv2.putText(out, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def load_pt_model(weights: pathlib.Path, yolov5_dir: pathlib.Path):
    import pathlib as pathlib_module
    import torch

    pathlib_module.WindowsPath = pathlib_module.PosixPath
    sys.path.insert(0, str(yolov5_dir))
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    return (ckpt.get("ema") or ckpt["model"]).float().eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--yolov5-dir", default="duo-tpu/workspace/traffic_light_crosswalk_yolov5s/yolov5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    image_dir = pathlib.Path(args.images)
    out_dir = pathlib.Path(args.out)
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    pt_model = load_pt_model(pathlib.Path(args.pt), pathlib.Path(args.yolov5_dir))
    ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    all_details = []
    rows = []
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        inp, prep = to_input(image)
        with torch.no_grad():
            pt_pred = pt_model(torch.from_numpy(inp))[0].detach().cpu().numpy()
        onnx_outputs = ort_session.run(None, {input_name: inp})

        results = {
            "pt": decode_pt(pt_pred, prep, args.conf, args.iou, args.max_det),
            "onnx": decode_onnx_split(onnx_outputs, prep, args.conf, args.iou, args.max_det),
        }
        for model_name, dets in results.items():
            cv2.imwrite(str(vis_dir / f"{image_path.stem}_{model_name}.jpg"), draw(image, dets))
            counts = {name: 0 for name in CLASS_NAMES}
            for det in dets:
                counts[det["class_name"]] += 1
                all_details.append({
                    "image": image_path.name,
                    "model": model_name,
                    **det,
                })
            rows.append({
                "image": image_path.name,
                "model": model_name,
                "detections": len(dets),
                **counts,
            })

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "model", "detections", *CLASS_NAMES])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "detections.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"images={len(images)} rows={len(rows)} details={len(all_details)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
