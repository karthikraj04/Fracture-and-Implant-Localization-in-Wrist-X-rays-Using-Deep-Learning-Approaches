import os
import shutil
import random
from pathlib import Path

# Source directories
src_images = Path("final_dataset/images")
src_labels = Path("final_dataset/labels")

# Destination directories
dst_base = Path("yolo_dataset")
dst_train_images = dst_base / "images/train"
dst_val_images = dst_base / "images/val"
dst_train_labels = dst_base / "labels/train"
dst_val_labels = dst_base / "labels/val"

# Create directories
for d in [dst_train_images, dst_val_images, dst_train_labels, dst_val_labels]:
    d.mkdir(parents=True, exist_ok=True)

# Split files
all_files = list(src_labels.glob("*.txt"))
random.shuffle(all_files)

split_idx = int(0.8 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

def copy_files(files, dst_img, dst_lbl):
    for label_path in files:
        img_path = src_images / (label_path.stem + ".png")
        if img_path.exists():
            shutil.copy(img_path, dst_img)
            shutil.copy(label_path, dst_lbl)

copy_files(train_files, dst_train_images, dst_train_labels)
copy_files(val_files, dst_val_images, dst_val_labels)

print(f"✅ Split complete: {len(train_files)} train, {len(val_files)} val")
