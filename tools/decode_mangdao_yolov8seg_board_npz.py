#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xywh_to_xyxy(boxes):
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes, scores, iou_threshold):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = np.maximum(0.0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0.0, boxes[i, 3] - boxes[i, 1])
        area_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def letterbox_info(width, height, target=640):
    ratio = min(target / width, target / height)
    resized_w = max(1, int(width * ratio))
    resized_h = max(1, int(height * ratio))
    paste_x = (target - resized_w) // 2
    paste_y = (target - resized_h) // 2
    return ratio, resized_w, resized_h, paste_x, paste_y


def crop_mask(mask, box, mask_size=160, input_size=640):
    scale = mask_size / input_size
    x1 = int(np.floor(box[0] * scale))
    y1 = int(np.floor(box[1] * scale))
    x2 = int(np.ceil(box[2] * scale))
    y2 = int(np.ceil(box[3] * scale))
    x1 = int(np.clip(x1, 0, mask_size))
    y1 = int(np.clip(y1, 0, mask_size))
    x2 = int(np.clip(x2, 0, mask_size))
    y2 = int(np.clip(y2, 0, mask_size))

    cropped = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def decode_one(npz_path, image_path, output_image, output_json, conf, iou, max_det):
    data = np.load(npz_path)
    names = sorted(data.files)
    det_name = next(n for n in names if data[n].size == 310800)
    proto_name = next(n for n in names if data[n].size == 819200)

    det_raw = data[det_name]
    proto_raw = data[proto_name]
    det_scale = 1.0 if det_raw.dtype == np.float32 else 0.702112
    proto_scale = 1.0 if proto_raw.dtype == np.float32 else 0.0476352
    det = det_raw.astype(np.float32).reshape(1, 37, 8400, 1).squeeze((0, 3)) * det_scale
    proto = proto_raw.astype(np.float32).reshape(1, 32, 160, 160).squeeze(0) * proto_scale

    pred = det.T
    boxes_640 = xywh_to_xyxy(pred[:, :4])
    scores = pred[:, 4]
    coeffs = pred[:, 5:37]

    candidates = np.where(scores >= conf)[0]
    boxes_640 = boxes_640[candidates]
    scores = scores[candidates]
    coeffs = coeffs[candidates]

    if boxes_640.size:
        keep = nms(boxes_640, scores, iou)[:max_det]
        boxes_640 = boxes_640[keep]
        scores = scores[keep]
        coeffs = coeffs[keep]

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {image_path}")
    height, width = image.shape[:2]
    ratio, resized_w, resized_h, paste_x, paste_y = letterbox_info(width, height)

    overlay = image.copy()
    proto_flat = proto.reshape(32, -1)
    rows = []
    for idx, (box, score, coeff) in enumerate(zip(boxes_640, scores, coeffs)):
        mask160 = sigmoid(coeff @ proto_flat).reshape(160, 160)
        mask160 = crop_mask(mask160, box)
        mask640 = cv2.resize(mask160, (640, 640), interpolation=cv2.INTER_LINEAR)
        mask_crop = mask640[paste_y:paste_y + resized_h, paste_x:paste_x + resized_w]
        mask_orig = cv2.resize(mask_crop, (width, height), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask_orig > 0.5

        color = np.array([255, 80, 40], dtype=np.uint8)
        overlay[mask_bool] = (overlay[mask_bool] * 0.45 + color * 0.55).astype(np.uint8)

        x1 = (box[0] - paste_x) / ratio
        y1 = (box[1] - paste_y) / ratio
        x2 = (box[2] - paste_x) / ratio
        y2 = (box[3] - paste_y) / ratio
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(overlay, f"tactile_paving {score:.2f}", (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append({
            "class_id": 0,
            "class_name": "tactile_paving",
            "confidence": float(score),
            "xyxy": [x1, y1, x2, y2],
            "mask_pixels": int(mask_bool.sum()),
        })

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), overlay)
    output_json.write_text(json.dumps({
        "image": str(image_path),
        "npz": str(npz_path),
        "count": len(rows),
        "detections": rows,
    }, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=20)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    json_dir = out_dir / "json"
    summary = []

    for npz_path in sorted(raw_dir.glob("*_board_out.npz")):
        stem = npz_path.name.replace("_board_out.npz", "")
        rows = decode_one(
            npz_path,
            image_dir / f"{stem}.jpg",
            vis_dir / f"{stem}_board_seg.jpg",
            json_dir / f"{stem}_board_seg.json",
            args.conf,
            args.iou,
            args.max_det,
        )
        for i, row in enumerate(rows):
            summary.append([stem + ".jpg", i, row["class_name"], f"{row['confidence']:.6f}", row["mask_pixels"]])
        if not rows:
            summary.append([stem + ".jpg", "", "", "", ""])

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "detection_index", "class", "confidence", "mask_pixels"])
        writer.writerows(summary)


if __name__ == "__main__":
    main()
