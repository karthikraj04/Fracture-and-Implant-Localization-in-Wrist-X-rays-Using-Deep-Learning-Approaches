import os
import shutil
from tqdm import tqdm

# --------- INPUT PATHS ---------
# Healthy + Fracture (already combined, 0 or empty)
HF_IMAGES_TRAIN = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\yolov8_dataset\images\train"
HF_IMAGES_VAL   = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\yolov8_dataset\images\val"
HF_LABELS_TRAIN = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\yolov8_dataset\labels\train"
HF_LABELS_VAL   = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\yolov8_dataset\labels\val"

# Implant (remapped to class 2 already)
INPLANT_IMAGES = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\datasets\implants\images"
INPLANT_LABELS = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\datasets\implants\labels"

# --------- OUTPUT PATHS ---------
MERGE_ROOT = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope\merged_dataset"
MERGE_IMG_TRAIN = os.path.join(MERGE_ROOT, "images", "train")
MERGE_IMG_VAL   = os.path.join(MERGE_ROOT, "images", "val")
MERGE_LBL_TRAIN = os.path.join(MERGE_ROOT, "labels", "train")
MERGE_LBL_VAL   = os.path.join(MERGE_ROOT, "labels", "val")

# Create dirs
for d in [MERGE_IMG_TRAIN, MERGE_IMG_VAL, MERGE_LBL_TRAIN, MERGE_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

def copy_set(image_dir, label_dir, dst_img_dir, dst_lbl_dir):
    count = 0
    for fname in tqdm(os.listdir(image_dir), desc=f"Copying {os.path.basename(dst_img_dir)}"):
        src_img = os.path.join(image_dir, fname)
        src_lbl = os.path.join(label_dir, fname.replace(".png", ".txt").replace(".jpg", ".txt"))

        dst_img = os.path.join(dst_img_dir, fname)
        dst_lbl = os.path.join(dst_lbl_dir, os.path.basename(src_lbl))

        if not os.path.exists(src_img):
            continue

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()  # empty label = healthy

        count += 1
    return count

# Copy fracture + healthy (already class 0 or empty)
hf_train = copy_set(HF_IMAGES_TRAIN, HF_LABELS_TRAIN, MERGE_IMG_TRAIN, MERGE_LBL_TRAIN)
hf_val   = copy_set(HF_IMAGES_VAL,   HF_LABELS_VAL,   MERGE_IMG_VAL,   MERGE_LBL_VAL)

# Copy implant (already remapped to class 2)
implant_trainval = 0
for fname in tqdm(os.listdir(INPLANT_IMAGES), desc="Copying Implants"):
    src_img = os.path.join(INPLANT_IMAGES, fname)
    src_lbl = os.path.join(INPLANT_LABELS, fname.replace(".png", ".txt").replace(".jpg", ".txt"))

    dst_img_dir = MERGE_IMG_TRAIN if implant_trainval % 5 else MERGE_IMG_VAL
    dst_lbl_dir = MERGE_LBL_TRAIN if implant_trainval % 5 else MERGE_LBL_VAL

    shutil.copy2(src_img, os.path.join(dst_img_dir, fname))
    shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, os.path.basename(src_lbl)))

    implant_trainval += 1

# ✅ Summary
print(f"\n✅ Merged healthy + fracture: {hf_train} train, {hf_val} val")
print(f"✅ Merged implants: {implant_trainval - implant_trainval // 5} train, {implant_trainval // 5} val")
