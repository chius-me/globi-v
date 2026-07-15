#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms


PROFILE = {
    "input_size": (640, 640),
    "reg_max": 16,
    "nc": 3,
    "strides": [8, 16, 32],
    "class_names": ["red_light", "green_light", "yellow_light"],
    "reg_heads": [
        ("output0_Conv", 0.197428),
        ("367_Conv", 0.22775),
        ("374_Conv", 0.102542),
    ],
    "cls_heads": [
        ("381_Conv", 0.523613),
        ("388_Conv", 0.472823),
        ("395_Conv", 0.188022),
    ],
}

COLORS = {
    0: (0, 0, 255),
    1: (0, 200, 0),
    2: (0, 215, 255),
}


def parse_args():
    p = argparse.ArgumentParser(description="Decode CVI runtime YOLOv8-style 6-output npz and draw boxes.")
    p.add_argument("--image", required=True, help="Original input image path used for inference")
    p.add_argument("--npz", required=True, help="Output npz from cvi_minimal_runner")
    p.add_argument("--save", required=True, help="Path to save annotated image")
    p.add_argument("--save-json", default="", help="Optional path to save decoded detections as JSON")
    p.add_argument("--conf", type=float, default=0.25, help="Global confidence threshold after sigmoid")
    p.add_argument("--classwise-conf", default="", help="Optional per-class confidence thresholds, e.g. 'yellow_light=0.40' or 'red_light=0.25,green_light=0.25,yellow_light=0.40'")
    p.add_argument("--iou", type=float, default=0.50, help="IoU threshold for NMS")
    p.add_argument("--max-det", type=int, default=100, help="Max detections to keep")
    p.add_argument("--stretch", action="store_true", help="Use direct resize inverse mapping instead of letterbox")
    return p.parse_args()


def make_anchor_grid(h: int, w: int, stride: int):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    points = torch.stack((xs + 0.5, ys + 0.5), dim=-1).view(-1, 2)
    strides = torch.full((h * w, 1), float(stride), dtype=torch.float32)
    return points, strides


def dfl_decode(reg_tensor: torch.Tensor, reg_max: int):
    b, _, anchors = reg_tensor.shape
    proj = torch.arange(reg_max, dtype=reg_tensor.dtype, device=reg_tensor.device)
    reg_tensor = reg_tensor.view(b, 4, reg_max, anchors).transpose(2, 1)
    reg_tensor = reg_tensor.softmax(1)
    return (reg_tensor * proj.view(1, reg_max, 1, 1)).sum(1)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor):
    lt, rb = distance.chunk(2, dim=1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    cxy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((cxy, wh), dim=1)


def xywh2xyxy(x: torch.Tensor):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def load_predictions(npz_path: Path):
    z = np.load(npz_path)
    reg_all = []
    cls_all = []
    anchor_all = []
    stride_all = []

    for (reg_name, reg_scale), (cls_name, cls_scale), stride in zip(PROFILE["reg_heads"], PROFILE["cls_heads"], PROFILE["strides"]):
        reg = z[reg_name].astype(np.float32) * reg_scale
        cls = z[cls_name].astype(np.float32) * cls_scale
        _, _, h, w = reg.shape
        reg_t = torch.from_numpy(reg.reshape(1, reg.shape[1], -1))
        cls_t = torch.from_numpy(cls.reshape(1, cls.shape[1], -1))
        reg_all.append(reg_t)
        cls_all.append(cls_t)
        anchors, strides = make_anchor_grid(h, w, stride)
        anchor_all.append(anchors)
        stride_all.append(strides)

    reg_cat = torch.cat(reg_all, dim=2)
    cls_cat = torch.cat(cls_all, dim=2)
    anchors = torch.cat(anchor_all, dim=0).T.unsqueeze(0)
    strides = torch.cat(stride_all, dim=0).T.unsqueeze(0)

    dbox = dist2bbox(dfl_decode(reg_cat, PROFILE["reg_max"]), anchors) * strides
    pred = torch.cat((dbox, cls_cat.sigmoid()), dim=1)  # [1, 4 + nc, A]
    return pred


def build_classwise_thresholds(global_conf: float, classwise_conf_text: str):
    thresholds = {name: float(global_conf) for name in PROFILE["class_names"]}
    if not classwise_conf_text:
        return thresholds

    valid_names = set(PROFILE["class_names"])
    for item in classwise_conf_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --classwise-conf item: {item!r}")
        class_name, raw_value = item.split("=", 1)
        class_name = class_name.strip()
        raw_value = raw_value.strip()
        if class_name not in valid_names:
            raise ValueError(f"unknown class in --classwise-conf: {class_name}")
        value = float(raw_value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"classwise confidence must be between 0 and 1: {item}")
        thresholds[class_name] = value
    return thresholds


def multiclass_nms(pred: torch.Tensor, conf_thres: float, iou_thres: float, max_det: int, classwise_conf=None):
    pred = pred[0].transpose(0, 1)  # [A, 4+nc]
    boxes = xywh2xyxy(pred[:, :4])
    class_scores = pred[:, 4:]
    conf, cls_idx = class_scores.max(dim=1)
    if classwise_conf:
        per_anchor_thresholds = torch.tensor(
            [classwise_conf[PROFILE["class_names"][int(c)]] for c in cls_idx.tolist()],
            dtype=conf.dtype,
            device=conf.device,
        )
        keep = conf > per_anchor_thresholds
    else:
        keep = conf > conf_thres
    boxes = boxes[keep]
    conf = conf[keep]
    cls_idx = cls_idx[keep]
    if boxes.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32)

    offsets = cls_idx.to(boxes.dtype).unsqueeze(1) * 7680.0
    keep_idx = nms(boxes + offsets.repeat(1, 4), conf, iou_thres)[:max_det]
    return torch.cat((boxes[keep_idx], conf[keep_idx, None], cls_idx[keep_idx, None].to(torch.float32)), dim=1)


def undo_preprocess_xyxy(boxes: np.ndarray, orig_w: int, orig_h: int, stretch: bool):
    input_w, input_h = PROFILE["input_size"]
    out = boxes.copy().astype(np.float32)
    if stretch:
        scale_x = input_w / orig_w
        scale_y = input_h / orig_h
        out[:, [0, 2]] /= scale_x
        out[:, [1, 3]] /= scale_y
    else:
        ratio = min(input_w / orig_w, input_h / orig_h)
        resized_w = max(1, int(orig_w * ratio))
        resized_h = max(1, int(orig_h * ratio))
        pad_x = (input_w - resized_w) / 2.0
        pad_y = (input_h - resized_h) / 2.0
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / ratio
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h - 1)
    return out


def draw_boxes(image: np.ndarray, dets: np.ndarray):
    canvas = image.copy()
    for x1, y1, x2, y2, conf, cls_id in dets:
        cls_id = int(cls_id)
        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{PROFILE['class_names'][cls_id]} {conf:.3f}"
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, p1, p2, color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, p1[1] - th - 8)
        label_y2 = label_y1 + th + 8
        cv2.rectangle(canvas, (p1[0], label_y1), (p1[0] + tw + 8, label_y2), color, -1)
        cv2.putText(canvas, label, (p1[0] + 4, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    image_path = Path(args.image)
    npz_path = Path(args.npz)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)

    classwise_conf = build_classwise_thresholds(args.conf, args.classwise_conf)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    orig_h, orig_w = image.shape[:2]

    pred = load_predictions(npz_path)
    dets = multiclass_nms(pred, args.conf, args.iou, args.max_det, classwise_conf=classwise_conf).cpu().numpy()
    if len(dets):
        dets[:, :4] = undo_preprocess_xyxy(dets[:, :4], orig_w, orig_h, args.stretch)
        dets = dets[np.argsort(-dets[:, 4])]

    vis = draw_boxes(image, dets)
    cv2.imwrite(str(save_path), vis)

    detections = []
    for x1, y1, x2, y2, conf, cls_id in dets:
        detections.append(
            {
                "class_id": int(cls_id),
                "class_name": PROFILE["class_names"][int(cls_id)],
                "confidence": float(conf),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    report = {
        "image": str(image_path),
        "npz": str(npz_path),
        "save": str(save_path),
        "count": len(detections),
        "conf": args.conf,
        "classwise_conf": classwise_conf,
        "iou": args.iou,
        "detections": detections,
    }
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
