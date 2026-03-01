import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import time

# -------------------------
# Load YOLO model from analysis folder
# -------------------------
MODEL_PATH = "analysis/best.pt"
model = YOLO(MODEL_PATH)

# -------------------------
# Streamlit App Title
# -------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("YOLOv8 Advanced Object Detection Dashboard")
st.markdown("**Detect Vehicles, People, and Household Items**")

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Navigation")
options = ["Detect Image", "Detect Video", "Metrics & Graphs"]
choice = st.sidebar.radio("Choose Option", options)

# -------------------------
# Confidence Slider
# -------------------------
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# -------------------------
# Image Detection Function
# -------------------------
def detect_image(uploaded_file):
    image = Image.open(uploaded_file)
    results = model(image, conf=confidence)
    st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
    
    st.write("**Detected Classes and Confidence:**")
    st.dataframe(results.pandas().xywh[0][["name", "confidence"]])

# -------------------------
# Video Detection Function
# -------------------------
def detect_video(uploaded_file):
    tfile = uploaded_file
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_path = "output_video.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    stframe = st.empty()
    progress_text = "Processing video..."
    my_bar = st.progress(0, text=progress_text)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
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

# -------------------------
# Metrics & Graphs Function
# -------------------------
def display_metrics():
    st.subheader("Model Metrics and Training Graphs")
    try:
        df = pd.read_csv("analysis/results.csv")
        st.subheader("mAP over Epochs")
        st.line_chart(df[['metrics/mAP50','metrics/mAP50-95']])
        
        st.subheader("Precision-Recall Curve")
        st.image("analysis/PR_curve.png", use_column_width=True)
        
        st.subheader("Loss Curves")
        st.image("analysis/results.png", use_column_width=True)
        
        st.subheader("Confusion Matrix")
        st.image("analysis/confusion_matrix.png", use_column_width=True)
        
    except Exception as e:
        st.error("Metrics files not found in analysis/ folder!")
        st.write(e)

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
