import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import base64
import os

# -------------------------
# Load YOLO model
# -------------------------
MODEL_PATH = "analysis/best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please upload your trained weights.")
else:
    model = YOLO(MODEL_PATH)

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("YOLOv8 Object Detection Dashboard")
st.markdown("**Detect Vehicles, People, and Household Items**")

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Navigation")
options = ["Detect Image", "Detect Video", "Metrics & Graphs", "Failure Cases"]
choice = st.sidebar.radio("Choose Option", options)

# -------------------------
# Confidence Slider
# -------------------------
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# -------------------------
# Helper function to download files
# -------------------------
def download_file(file_path, file_label):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}">Download {file_label}</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning(f"{file_label} not found!")

# -------------------------
# Image Detection
# -------------------------
def detect_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        results = model(image, conf=confidence)
        st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
        
        st.write("**Detected Classes and Confidence:**")
        st.dataframe(results.pandas().xywh[0][["name", "confidence"]])
        
        # Save and download
        results.save("analysis/temp_image_result.png")
        download_file("analysis/temp_image_result.png", "prediction.png")
    except Exception as e:
        st.error("Error during image detection!")
        st.write(e)

# -------------------------
# Video Detection
# -------------------------
def detect_video(uploaded_file):
    try:
        tfile = uploaded_file
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out_path = "analysis/output_video.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        stframe = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        progress_text = "Processing video..."
        my_bar = st.progress(0, text=progress_text)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence)
            frame_out = results[0].plot()
            out.write(frame_out)
            stframe.image(frame_out, channels="BGR", use_column_width=True)
            
            frame_count += 1
            my_bar.progress(frame_count / total_frames, text=progress_text)
        
        cap.release()
        out.release()
        st.success("Video processed successfully!")
        st.video(out_path)
        download_file(out_path, "prediction_video.mp4")
    except Exception as e:
        st.error("Error during video detection!")
        st.write(e)

# -------------------------
# Metrics & Graphs
# -------------------------
def display_metrics():
    st.subheader("Model Metrics and Training Graphs")
    try:
        df_path = "analysis/results.csv"
        if not os.path.exists(df_path):
            raise FileNotFoundError("results.csv not found!")
        
        df = pd.read_csv(df_path)
        st.subheader("mAP over Epochs")
        st.line_chart(df[['metrics/mAP50','metrics/mAP50-95']])
        
        st.subheader("Loss Curve")
        loss_path = "analysis/loss_curve.png"
        if os.path.exists(loss_path):
            st.image(loss_path, use_column_width=True)
        else:
            st.warning("Loss curve image not found!")
        
        st.subheader("Precision-Recall Curve")
        pr_path = "analysis/PR_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
        else:
            st.warning("PR curve image not found!")
        
        st.subheader("Confusion Matrix")
        cm_path = "analysis/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.warning("Confusion matrix image not found!")
        
        st.subheader("Dataset Distribution")
        for fname in ["class_distribution_bar.png","class_distribution_pie.png","example_images.png"]:
            fpath = f"analysis/{fname}"
            if os.path.exists(fpath):
                st.image(fpath, use_column_width=True)
            else:
                st.warning(f"{fname} not found!")
    except Exception as e:
        st.error("Error displaying metrics!")
        st.write(e)

# -------------------------
# Failure Cases / Error Analysis
# -------------------------
def failure_cases():
    st.subheader("Failure Cases / Error Analysis")
    for idx, fname in enumerate(["failure1.png","failure2.png"], start=1):
        fpath = f"analysis/{fname}"
        if os.path.exists(fpath):
            st.image(fpath, caption=f"Example {idx}")
        else:
            st.warning(f"{fname} not found!")
    st.markdown("""
    **Notes / Possible Improvements:**
    - Add more images for rare classes  
    - Apply data augmentation  
    - Fine-tune confidence threshold per class  
    - Increase epochs or try different YOLO architectures
    """)

# -------------------------
# Main Navigation
# -------------------------
if choice == "Detect Image":
    st.sidebar.subheader("Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_image:
        detect_image(uploaded_image)

elif choice == "Detect Video":
    st.sidebar.subheader("Upload Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4","avi"])
    if uploaded_video:
        detect_video(uploaded_video)

elif choice == "Metrics & Graphs":
    display_metrics()

elif choice == "Failure Cases":
    failure_cases()
