#!/usr/bin/env python3
"""Convert LISA Traffic Light Dataset (CSV format) to YOLO format.
Classes: 0=red_light(stop), 1=green_light(go), 2=yellow_light(warning)
"""

import os, csv, sys, shutil
from pathlib import Path
from collections import Counter

# LISA tag → YOLO class mapping
TAG_MAP = {
    # Red / Stop lights
    'stop': 0, 'red': 0,
    # Green / Go lights
    'go': 1, 'green': 1,
    # Yellow / Warning lights
    'warning': 2, 'yellow': 2,
}

def main(lisa_root, output_dir):
    lisa_root = Path(lisa_root).resolve()
    output_dir = Path(output_dir).resolve()
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all CSV annotation files
    csv_dir = lisa_root / 'Annotations' / 'Annotations'
    csv_files = list(csv_dir.rglob('frameAnnotationsBOX.csv'))
    print(f"Found {len(csv_files)} annotation CSV files")
    
    # Map: image_filename → [(class, x_center, y_center, w, h)]
    image_anns = {}
    image_dims = {}  # filename → (width, height) - needed for normalization
    
    # First pass: collect all annotations and image dimensions
    for csv_path in csv_files:
        # Infer source: dayTrain, nightTrain, daySequence1, daySequence2, nightSequence1, nightSequence2
        csv_parent = str(csv_path.parent).lower()
        if 'night' in csv_parent:
            subset = 'night'
        else:
            subset = 'day'
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                tag = row['Annotation tag'].strip().lower()
                if tag not in TAG_MAP:
                    continue
                
                cls = TAG_MAP[tag]
                # CSV filename is like "dayTraining/dayClip1--00000.jpg"
                filename = row['Filename'].strip()
                # Normalize: dayTraining/ → dayTrain/, daySequence1/ → keep
                
                xmin = float(row['Upper left corner X'])
                ymin = float(row['Upper left corner Y'])
                xmax = float(row['Lower right corner X'])
                ymax = float(row['Lower right corner Y'])
                
                # Find the actual image file
                img_path = find_image(lisa_root, filename)
                if img_path is None:
                    continue
                
                # Get image dimensions
                if filename not in image_dims:
                    import cv2
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    image_dims[filename] = (w, h)
                
                w_img, h_img = image_dims[filename]
                
                # Convert to YOLO format
                x_c = ((xmin + xmax) / 2) / w_img
                y_c = ((ymin + ymax) / 2) / h_img
                b_w = (xmax - xmin) / w_img
                b_h = (ymax - ymin) / h_img
                
                # Validate
                if b_w <= 0 or b_h <= 0 or b_w > 1 or b_h > 1:
                    continue
                
                if filename not in image_anns:
                    image_anns[filename] = []
                image_anns[filename].append({
                    'cls': cls,
                    'img_path': img_path,
                    'yolo': (x_c, y_c, b_w, b_h),
                    'subset': subset,
                })
    
    print(f"Parsed {sum(len(v) for v in image_anns.values())} annotations across {len(image_anns)} images")
    
    # Class distribution
    cls_counts = Counter()
    for anns in image_anns.values():
        for a in anns:
            cls_counts[a['cls']] += 1
    cls_names = {0: 'red_light', 1: 'green_light', 2: 'yellow_light'}
    print(f"Class distribution: {{{', '.join(f'{cls_names[k]}: {v}' for k, v in sorted(cls_counts.items()))}}}")
    
    if len(image_anns) < 20:
        print("ERROR: Too few annotated images!")
        return
    
    # Split: stratify by clip for better val distribution, keep day/night balanced
    # Simple approach: 80/20 random split
    import random
    random.seed(42)
    filenames = list(image_anns.keys())
    random.shuffle(filenames)
    split_idx = int(len(filenames) * 0.85)
    train_fns = set(filenames[:split_idx])
    val_fns = set(filenames[split_idx:])
    
    splits = {'train': train_fns, 'val': val_fns}
    
    # Write output
    img_counter = 0
    for split_name, fn_set in splits.items():
        for fn in fn_set:
            anns = image_anns[fn]
            first = anns[0]
            img_path = first['img_path']
            
            # Use short unique name
            new_name = f"{fn.replace('/', '_').replace('--', '_')}"
            dst_img = output_dir / 'images' / split_name / new_name
            dst_label = output_dir / 'labels' / split_name / new_name.replace('.jpg', '.txt').replace('.png', '.txt')
            
            shutil.copy2(img_path, dst_img)
            
            with open(dst_label, 'w') as f:
                for a in anns:
                    x, y, w, h = a['yolo']
                    f.write(f"{a['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            img_counter += 1
        
        print(f"  {split_name}: {len(fn_set)} images")
    
    print(f"Total exported: {img_counter} images")
    
    # Write dataset.yaml
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
nc: 3
names:
  0: red_light
  1: green_light
  2: yellow_light
"""
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDone! Output: {output_dir}")
    print(f"YAML: {output_dir / 'dataset.yaml'}")


def find_image(lisa_root, csv_filename):
    """Find the actual image file for a given CSV filename."""
    # csv_filename examples: 
    #   "dayTraining/dayClip1--00000.jpg" 
    #   "daySequence1/daySequence1--00000.jpg"
    
    parts = csv_filename.replace('\\', '/').split('/')
    
    # Try multiple possible paths
    candidates = []
    basename = parts[-1]
    
    # Direct path from root
    candidates.append(lisa_root / csv_filename)
    
    # LISA has: dayTrain/dayTrain/dayClip1/frames/...
    # CSV has: dayTraining/dayClip1--00000.jpg
    # So: dayTrain → dayTrain/dayTrain/dayClip1/frames/
    for prefix_map in [
        ('dayTraining', 'dayTrain/dayTrain'),
        ('nightTraining', 'nightTrain/nightTrain'),
        ('daySequence1', 'daySequence1/daySequence1'),
        ('daySequence2', 'daySequence2/daySequence2'),
        ('nightSequence1', 'nightSequence1/nightSequence1'),
        ('nightSequence2', 'nightSequence2/nightSequence2'),
    ]:
        if parts[0].startswith(prefix_map[0].split('/')[0]):
            # Find clip dir
            clip = parts[0]  # e.g., dayTraining
            for sub in lisa_root.rglob(f'{prefix_map[1]}/**/frames/{basename}'):
                return sub
            # Also try without frames subdir
            for sub in lisa_root.rglob(f'{prefix_map[1]}/**/{basename}'):
                return sub
    
    # Brute force search as fallback
    for p in lisa_root.rglob(basename):
        return p
    
    return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: convert_lisa_csv.py <lisa_root> [output_dir]")
        sys.exit(1)
    lisa_root = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './traffic_light_yolo'
    main(lisa_root, output_dir)
