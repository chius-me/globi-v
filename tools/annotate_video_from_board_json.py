#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2


COLORS = {
    "red_light": (0, 0, 255),
    "yellow_light": (0, 215, 255),
    "green_light": (0, 200, 0),
    "off": (160, 160, 160),
}


def draw(frame, detections):
    for det in detections:
        cls = det.get("class_name", "unknown")
        conf = float(det.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(round(v)) for v in det.get("xyxy", [0, 0, 0, 0])]
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.3f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, y0), (x1 + tw + 8, y0 + th + 8), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 4, y0 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--out-video", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    json_dir = Path(args.json_dir)
    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"failed to open output video: {out_video}")

    rows = []
    frame_idx = 1
    total_dets = 0
    hit_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        json_path = json_dir / f"frame_{frame_idx:06d}.json"
        detections = []
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                detections = json.load(f).get("detections", [])
        if detections:
            hit_frames += 1
            total_dets += len(detections)
        draw(frame, detections)
        writer.write(frame)
        rows.append({
            "frame": frame_idx,
            "count": len(detections),
            "detections": "; ".join(
                f'{d.get("class_name")}:{float(d.get("confidence", 0.0)):.3f}'
                for d in detections
            ),
        })
        frame_idx += 1

    cap.release()
    writer.release()

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["frame", "count", "detections"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"wrote {out_video}")
    print(f"frames={len(rows)} hit_frames={hit_frames} detections={total_dets}")


if __name__ == "__main__":
    main()
