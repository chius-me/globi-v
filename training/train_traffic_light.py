#!/usr/bin/env python3
"""Train YOLOv5s on traffic light dataset for Globi-V smart guide cane.

Base model: YOLOv5s — winner of full 13-model DuoS benchmark (738ms, 1.36 FPS).
"""

from ultralytics import YOLO
import os

# Config
DATASET_YAML = os.path.expanduser("~/repo/github/globi/training/traffic_light_yolo/dataset.yaml")
MODEL_NAME = "yolov5s.pt"
PROJECT = os.path.expanduser("~/repo/github/globi/training/runs")
NAME = "traffic_light_yolov5s"

print(f"Dataset: {DATASET_YAML}")
print(f"Base model: {MODEL_NAME}")
print(f"Project: {PROJECT}/{NAME}")

# Load pretrained model
model = YOLO(MODEL_NAME)

# Train
results = model.train(
    data=DATASET_YAML,
    epochs=50,
    imgsz=640,
    batch=8,           # RTX 4060 8GB should handle this
    device=0,
    workers=2,
    project=PROJECT,
    name=NAME,
    patience=10,         # early stopping
    save=True,
    save_period=10,
    val=True,
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    # Augmentation (moderate for small dataset)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)

# Validate
print("\n=== Validation ===")
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Export to ONNX for TPU-MLIR
print("\n=== Export ONNX ===")
export_path = model.export(format='onnx', imgsz=640, opset=11, simplify=True)
print(f"ONNX saved: {export_path}")

print("\n=== Training Complete ===")
print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")
print(f"ONNX model: {export_path}")
