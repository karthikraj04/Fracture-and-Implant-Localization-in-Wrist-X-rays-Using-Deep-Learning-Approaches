import os
import cv2
import numpy as np

# Set both paths
input_dirs = {
    "negative": "fracture-gan/data/wrist_negative",
    "positive": "fracture-gan/data/wrist_positive"
}
output_base = "fracture-gan/data/processed"
target_size = (256, 256)

for label, input_dir in input_dirs.items():
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.endswith(".png"):
            path = os.path.join(input_dir, img_name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"⚠️ Skipped unreadable: {path}")
                continue

            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 127.5 - 1.0

            out_path = os.path.join(output_dir, img_name.replace('.png', '.npy'))
            np.save(out_path, img)
            print(f"✅ Saved: {out_path}")
