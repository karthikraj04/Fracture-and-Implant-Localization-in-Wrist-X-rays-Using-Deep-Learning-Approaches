import os
import shutil

# Use absolute paths to avoid FileNotFoundError
base_dir = r"C:\Users\ASUS\OneDrive\Desktop\Final_hope"
images_dir = os.path.join(base_dir, "annotated_dataset", "images")
labels_dir = os.path.join(base_dir, "annotated_dataset", "labels")

# Destination folder
output_dir = os.path.join(base_dir, "final_dataset")
output_images = os.path.join(output_dir, "images")
output_labels = os.path.join(output_dir, "labels")

# Create output folders if they don't exist
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

count = 0
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        label_path = os.path.join(labels_dir, label_file)
        image_name = os.path.splitext(label_file)[0] + ".png"
        image_path = os.path.join(images_dir, image_name)

        if os.path.exists(image_path):
            shutil.copy(image_path, output_images)
            shutil.copy(label_path, output_labels)
            count += 1

print(f"✅ Copied {count} labeled images and labels to final_dataset.")
