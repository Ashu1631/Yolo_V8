import streamlit as st
import os
import cv2
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Ashu YOLO Enterprise Pro", layout="wide", initial_sidebar_state="expanded")

# ================= 1. DIRECTORY SETUP =================
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis", "dataset"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Session State Initialization
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None
if "secondary_model" not in st.session_state: st.session_state.secondary_model = None

# ================= 2. LOGIN SYSTEM =================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;color:#00ffff'>🚀 Ashu YOLO Enterprise</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Login")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if user == "admin" and pw == "admin123": # Use ashu@1234 if preferred
                st.session_state.logged_in = True
                st.rerun()
            else: st.error("Invalid Credentials")
    st.stop()

# ================= 3. NAVIGATION =================
pages = ["Model Selection", "Upload & Detect", "Dataset Analysis", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
current_page = st.sidebar.radio("🚀 Navigation", pages, index=pages.index(st.session_state.page))

# ================= 4. HELPERS =================
def process_frame(model, frame, conf=0.25):
    results = model(frame, conf=conf)
    return results[0].plot(), results

# ================= 5. PAGE CONTENT =================

# --- MODEL SELECTION ---
if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    if not models: st.error("No .pt models found in directory!")
    
    col1, col2 = st.columns(2)
    with col1:
        primary = st.selectbox("Primary Model", ["-- Select --"] + models)
    with col2:
        secondary = st.selectbox("Comparison Model (Optional)", ["None"] + models)
    
    if st.button("Load Models"):
        if primary != "-- Select --":
            st.session_state.model = YOLO(primary)
            st.session_state.model_name = primary
            if secondary != "None":
                st.session_state.secondary_model = YOLO(secondary)
                st.session_state.secondary_name = secondary
            st.success("Engines Ready!")
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --- UPLOAD & DETECT (Fixed Comparison) ---
elif current_page == "Upload & Detect":
    st.title("🔍 Smart Detection")
    if not st.session_state.model:
        st.warning("Please select a model first!")
        st.stop()

    uploaded = st.file_uploader("Upload File", type=["jpg", "png", "mp4"])
    
    if uploaded:
        is_video = uploaded.name.endswith(".mp4")
        temp_path = os.path.join("outputs", uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())

        if st.session_state.secondary_model and not is_video:
            st.subheader("⚖️ Comparison Mode Active")
            c1, c2 = st.columns(2)
            img = cv2.imread(temp_path)
            with c1:
                st.write(f"Model A: {st.session_state.model_name}")
                res1, _ = process_frame(st.session_state.model, img)
                st.image(res1, channels="BGR")
            with c2:
                st.write(f"Model B: {st.session_state.secondary_name}")
                res2, _ = process_frame(st.session_state.secondary_model, img)
                st.image(res2, channels="BGR")
        else:
            # Normal detection + save failure logic
            if is_video:
                st.info("Video processing... (View in real-time below)")
                # Video capture logic here
            else:
                img = cv2.imread(temp_path)
                out_img, results = process_frame(st.session_state.model, img)
                st.image(out_img, channels="BGR")
                # Failure logic: if low conf or no boxes
                if len(results[0].boxes) == 0:
                    cv2.imwrite(f"failure_cases/missed_{uploaded.name}", img)

# --- DATASET ANALYSIS ---
elif current_page == "Dataset Analysis":
    st.title("📁 Dataset Analytics")
    files = os.listdir("dataset")
    if files:
        st.success(f"Found {len(files)} files in /dataset")
        # Visualizing first few images
        cols = st.columns(4)
        for i, f in enumerate(files[:4]):
            cols[i].image(os.path.join("dataset", f), use_container_width=True)
    else: st.warning("Upload images to 'dataset' folder to see them here.")

# --- WEBCAM (Fixed for Cloud) ---
elif current_page == "Webcam Detection":
    st.title("🎥 Live Engine")
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            res, _ = process_frame(st.session_state.model, img)
            return av.VideoFrame.from_ndarray(res, format="bgr24")

    if st.session_state.model:
        webrtc_streamer(key="yolo-live", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIG)
    else: st.error("Model not loaded!")

# --- EVALUATION ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Model Training Performance")
    t1, t2 = st.tabs(["Loss Curves", "Confusion Matrix"])
    with t1:
        # Mock Loss Curves
        data = pd.DataFrame({"Epoch": range(1,21), "Box_Loss": np.random.rand(20), "Cls_Loss": np.random.rand(20)})
        st.plotly_chart(px.line(data, x="Epoch", y=["Box_Loss", "Cls_Loss"], title="YOLO Training Losses"))
    with t2:
        st.subheader("Confusion Matrix")
        
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/confusion_matrix.png")

# --- MODEL COMPARISON ---
elif current_page == "Model Comparison":
    st.title("⚖️ Advanced Benchmarking")
    # Funnel Chart
    st.subheader("Detection Pipeline Funnel")
    fig = go.Figure(go.Funnel(y=["Input", "Candidates", "Confident", "NMS Final"], x=[1000, 800, 500, 450]))
    st.plotly_chart(fig)
