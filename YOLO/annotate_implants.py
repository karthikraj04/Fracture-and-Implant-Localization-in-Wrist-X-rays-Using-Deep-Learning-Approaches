import cv2
import os

INPUT_DIR = "MURA_selected/positive"
OUTPUT_IMG_DIR = "annotated_dataset/images"
OUTPUT_LABEL_DIR = "annotated_dataset/labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

bbox = []
drawing = False
ix, iy = -1, -1

def draw_rect(event, x, y, flags, param):
    global ix, iy, drawing, bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox.append([ix, iy, x, y])

def save_yolo_format(img_path, bboxes, label_path):
    h, w = cv2.imread(img_path).shape[:2]
    with open(label_path, 'w') as f:
        for box in bboxes:
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

image_list = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for img_name in image_list:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    bbox = []
    clone = img.copy()
    cv2.namedWindow("Draw Implant BBox - Press S to Save, N to Skip")
    cv2.setMouseCallback("Draw Implant BBox - Press S to Save, N to Skip", draw_rect)

    while True:
        temp = clone.copy()
        for box in bbox:
            cv2.rectangle(temp, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Draw Implant BBox - Press S to Save, N to Skip", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if bbox:
                # Save image
                out_img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
                out_label_path = os.path.join(OUTPUT_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
                cv2.imwrite(out_img_path, img)
                save_yolo_format(img_path, bbox, out_label_path)
            break
        elif key == ord('n'):
            break
    cv2.destroyAllWindows()
