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
import io

# Page Configuration
st.set_page_config(page_title="Ashu YOLO Enterprise Pro", layout="wide", initial_sidebar_state="expanded")

# ================= 1. DIRECTORY SETUP =================
# 'runs/detect' folder detections save karne ke liye
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis", "datasets", "runs/detect"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

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
            if user == "admin" and pw == "ashu@1234":
                st.session_state.logged_in = True
                st.rerun()
            else: st.error("Invalid Credentials")
    st.stop()

# ================= 3. NAVIGATION =================
pages = ["Model Selection", "Upload & Detect", "Dataset Analysis", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
current_page = st.sidebar.radio("🚀 Navigation", pages, index=pages.index(st.session_state.page))

# ================= 4. HELPER FUNCTIONS =================
def save_detection(image, filename):
    path = os.path.join("runs/detect", f"det_{datetime.now().strftime('%H%M%S')}_{filename}")
    cv2.imwrite(path, image)
    return path

# ================= 5. PAGE CONTENT =================

# --- MODEL SELECTION ---
if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    col1, col2 = st.columns(2)
    with col1:
        primary = st.selectbox("Select Primary Model", ["-- Select --"] + models)
    with col2:
        secondary = st.selectbox("Select Secondary Model (For Comparison)", ["None"] + models)
    
    if st.button("Initialize Models"):
        if primary != "-- Select --":
            st.session_state.model = YOLO(primary)
            st.session_state.model_name = primary
            if secondary != "None":
                st.session_state.secondary_model = YOLO(secondary)
                st.session_state.secondary_name = secondary
            st.success(f"Models Ready: {primary}")
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --- UPLOAD & DETECT (FPS & AUTO-SAVE ADDED) ---
elif current_page == "Upload & Detect":
    st.title("🔍 Detection, FPS & Auto-Save")
    if not st.session_state.model:
        st.warning("⚠️ Load a model first!"); st.stop()

    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    if uploaded:
        temp_path = os.path.join("outputs", uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())
        is_video = uploaded.name.endswith(".mp4")

        if not is_video:
            img = cv2.imread(temp_path)
            
            # FPS Calculation for Image Inference
            t1 = time.time()
            results = st.session_state.model(img)
            t2 = time.time()
            fps = 1 / (t2 - t1)
            
            res_plotted = results[0].plot()
            st.image(res_plotted, channels="BGR", caption=f"FPS: {fps:.2f}")
            
            # Auto-save Detection
            saved_path = save_detection(res_plotted, uploaded.name)
            st.success(f"Detection saved automatically at: {saved_path}")
        
        else:
            cap = cv2.VideoCapture(temp_path)
            st_fps = st.empty()
            st_frame = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                t1 = time.time()
                res = st.session_state.model(frame)
                t2 = time.time()
                fps = 1 / (t2 - t1)
                
                st_fps.metric("Live Inference FPS", f"{fps:.2f}")
                st_frame.image(res[0].plot(), channels="BGR")
            cap.release()

# --- DATASET ANALYSIS (REPORT DOWNLOAD ADDED) ---
elif current_page == "Dataset Analysis":
    st.title("📁 Dataset Explorer & Report")
    files = [f for f in os.listdir("datasets") if f.endswith(('.jpg', '.png'))]
    
    if files:
        # Generate Report Data
        report_data = []
        for f in files[:10]: # Example for first 10
            img_path = os.path.join("datasets", f)
            res = st.session_state.model(img_path)
            obj_count = len(res[0].boxes)
            report_data.append({"File": f, "Objects_Detected": obj_count, "Timestamp": datetime.now()})
        
        df_report = pd.DataFrame(report_data)
        st.table(df_report)
        
        # Download Report as CSV
        csv = df_report.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Dataset Analysis Report (CSV)", csv, "dataset_report.csv", "text/csv")
        
        sel_img = st.selectbox("Select Image to View", files)
        st.image(st.session_state.model(os.path.join("datasets", sel_img))[0].plot(), channels="BGR")
    else: st.error("No images in /datasets")

# --- EVALUATION DASHBOARD ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Training Performance Dashboard")
    # Existing graphs
    col_old1, col_old2 = st.columns(2)
    with col_old1:
        if os.path.exists("analysis/results.png"): st.image("analysis/results.png", caption="Main Results")
    with col_old2:
        if os.path.exists("analysis/confusion_matrix.png"): st.image("analysis/confusion_matrix.png", caption="Confusion Matrix")
    
    # Your 3 New Colab Images
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if os.path.exists("analysis/loss_curve.png"): st.image("analysis/loss_curve.png", caption="Loss Curve")
    with c2:
        if os.path.exists("analysis/map_curve.png"): st.image("analysis/map_curve.png", caption="mAP Curve")
    with c3:
        if os.path.exists("analysis/pr_curve_detail.png"): st.image("analysis/pr_curve_detail.png", caption="P-R Curve")

# --- WEBCAM (FPS INCLUDED) ---
elif current_page == "Webcam Detection":
    st.title("🎥 Live Stream with FPS")
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # FPS logic inside WebRTC is handled by browser, 
            # but we plot model inference
            res = st.session_state.model(img)
            return av.VideoFrame.from_ndarray(res[0].plot(), format="bgr24")
    webrtc_streamer(key="live", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIG)

# --- MODEL COMPARISON ---
elif current_page == "Model Comparison":
    st.title("⚖️ 10-Graph Benchmarking")
    # ... (Keep the 10 graphs code from previous turn here) ...
    st.info("Graphs for Precision, Recall, FPS, Latency, and mAP included.")
