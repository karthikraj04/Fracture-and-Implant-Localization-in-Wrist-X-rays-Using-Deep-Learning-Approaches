# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import torch
# import os
# import tempfile
# import cv2
# import numpy as np
# import sys



# # ✅ Ensure 'cbam' folder is in sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # ✅ Correct import (cbam.py is inside the cbam/ folder)


# # --- Title and Info ---
# st.set_page_config(page_title="Fracture Detection App", layout="centered")
# st.title("🦴 Fracture Detection from Wrist X-ray")
# st.markdown("Upload a wrist X-ray image to detect **Fractures** and **Implants** using a YOLOv8")

# # --- Load Model ---
# @st.cache_resource
# def load_model():
#     model_path = "C:/Users/Admin/Downloads/last.pt"

#     #model_path = "C:/Users/ASUS/OneDrive/Desktop/Final_hope/runs/detect/yolov8m_balanced_v1/weights/last.pt"
#     #model_path = "runs/detect/cbam_yolov8m_final3/weights/best.pt"
# # OR
# #   # if you want the last checkpoint

#     #model_path = "runs/detect/cbam_yolov8m_fast/weights/last.pt"

#     #model_path = "runs/cbam_yolov8m/weights/best.pt"
#     if not os.path.exists(model_path):
#         st.error("Model file not found. Please check the path.")
#         st.stop()

#     try:
#         model = YOLO(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         st.stop()

# model = load_model()

# # --- Upload Image ---
# uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display uploaded image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_container_width=True)


#     # Run Detection
#     if st.button("🔍 Detect Fracture / Implant"):
#         with st.spinner("Analyzing..."):
#             # Save to temp file for YOLO
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#                 image.save(tmp.name)
#                 results = model.predict(source=tmp.name, conf=0.25)

#             # Draw results
#             # Draw annotated image
#             annotated_frame = results[0].plot()
#             st.image(annotated_frame, caption="🩻 Detection Result", use_container_width=True)

#             # Classify result
#             pred_classes = [model.names[int(box.cls)] for box in results[0].boxes]  # e.g. ['fracture']
#             if "fracture" in pred_classes:
#                 st.markdown("### 🔴 **Fracture Detected**")
#             elif "implant" in pred_classes:
#                 st.markdown("### 🟡 **Implant Detected**")
#             elif "healthy" in pred_classes or len(pred_classes) == 0:
#                 st.markdown("### 🟢 **No Fracture or Implant Detected**")

#             # Optional download
#             result_path = "output_detected.jpg"
#             cv2.imwrite(result_path, annotated_frame[:, :, ::-1])  # Save RGB->BGR
#             with open(result_path, "rb") as file:
#                 st.download_button("💾 Download Result", file.read(), file_name="fracture_detection.jpg")

        
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import os
import tempfile
import cv2
import numpy as np
import sys

# ✅ Ensure 'cbam' folder is in sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Title and Info ---
st.set_page_config(page_title="Fracture Detection App", layout="centered")
st.title("🦴 Fracture Detection from Wrist X-ray")
st.markdown("Upload a wrist X-ray image to detect **Fractures**, **Implants**, or **Healthy** using YOLOv8.")

# --- Load Model ---
@st.cache_resource
def load_model():
    # Change this path to your actual model weights
    model_path = "C:/Users/Admin/Downloads/last.pt"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please check the path.")
        st.stop()

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model()

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📤 Uploaded Image", use_container_width=True)

    # Run Detection
    if st.button("🔍 Detect Fracture / Implant"):
        with st.spinner("Analyzing..."):
            # Save image temporarily for YOLO
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = model.predict(source=tmp.name, conf=0.25)

            # Get annotated frame
            annotated_frame = results[0].plot()

            with col2:
                st.image(annotated_frame, caption="🩻 Detection Result", use_container_width=True)

            # Extract predicted classes
            pred_classes = [model.names[int(box.cls)] for box in results[0].boxes]

            # Show classification result neatly below both columns
            if "fracture" in pred_classes:
                st.markdown("### 🔴 **Fracture Detected**")
            elif "implant" in pred_classes:
                st.markdown("### 🟡 **Implant Detected**")
            elif "healthy" in pred_classes or len(pred_classes) == 0:
                st.markdown("### 🟢 **No Fracture or Implant Detected**")

            # Allow download of result
            result_path = "output_detected.jpg"
            cv2.imwrite(result_path, annotated_frame[:, :, ::-1])  # Save RGB→BGR
            with open(result_path, "rb") as file:
                st.download_button("💾 Download Result", file.read(), file_name="fracture_detection.jpg")
