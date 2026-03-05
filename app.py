import streamlit as st
import pandas as pd
import os
import cv2
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
from datetime import datetime
import supervision as sv

# Page Configuration
st.set_page_config(page_title="Ashu YOLO Enterprise Pro", layout="wide", initial_sidebar_state="expanded")

# ================= 1. DIRECTORY SETUP =================
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis", "datasets"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None
if "secondary_model" not in st.session_state: st.session_state.secondary_model = None
    
# ================= 2. SLEEK PLOT HELPER =================
def get_sleek_plot(image, model):
    """Simple and Stable version that won't crash."""
    
    # 1. Standard Prediction (No complex tracking to avoid 'lap' errors)
    results = model(image, conf=0.3)[0]
    
    # 2. Get Detections
    detections = sv.Detections.from_ultralytics(results)
    
    # 3. Create simple labels with Percentage
    # model.names class index ko asli naam mein badal dega
    labels = [
        f"{model.names[class_id]} {confidence*100:.0f}%"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    
    # 4. Initialize Annotators (Basic version jo har jagah chalti hai)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.4,
        text_thickness=1,
        text_padding=4
    )
    
    # 5. Drawing
    # scene=image.copy() zaroori hai original image ko bachane ke liye
    annotated_frame = box_annotator.annotate(
        scene=image.copy(), 
        detections=detections
    )
    
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=labels
    )
    
    return annotated_frame
    
# ================= 3. LOGIN SYSTEM =================
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

# ================= 4. NAVIGATION =================
pages = ["Model Selection", "Upload & Detect", "Dataset Analysis", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
current_page = st.sidebar.radio("🚀 Navigation", pages, index=pages.index(st.session_state.page))

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
            st.success(f"Loaded: {primary}")
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --- UPLOAD & DETECT (With Model Name Labels) ---
elif current_page == "Upload & Detect":
    st.title("🔍 Sleek Detection & Comparison Hub")
    if not st.session_state.model:
        st.warning("⚠️ Load a model first!")
        st.stop()

    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    
    if uploaded:
        temp_path = os.path.join("outputs", uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())
        is_video = uploaded.name.endswith(".mp4")

        if st.session_state.secondary_model:
            c1, c2 = st.columns(2)
            c1.info(f"🟢 Model A: {st.session_state.model_name}")
            c2.info(f"🔵 Model B: {st.session_state.secondary_name}")
            
            if is_video:
                cap1, cap2 = cv2.VideoCapture(temp_path), cv2.VideoCapture(temp_path)
                out1, out2 = c1.empty(), c2.empty()
                while cap1.isOpened():
                    r1, f1 = cap1.read(); r2, f2 = cap2.read()
                    if not r1 or not r2: break
                    out1.image(get_sleek_plot(f1, st.session_state.model), channels="BGR")
                    out2.image(get_sleek_plot(f2, st.session_state.secondary_model), channels="BGR")
                cap1.release(); cap2.release()
            else:
                img = cv2.imread(temp_path)
                c1.image(get_sleek_plot(img, st.session_state.model), channels="BGR")
                c2.image(get_sleek_plot(img, st.session_state.secondary_model), channels="BGR")
        else:
            st.info(f"Using: {st.session_state.model_name}")
            if is_video:
                cap = cv2.VideoCapture(temp_path); out = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    out.image(get_sleek_plot(frame, st.session_state.model), channels="BGR")
            else:
                img = cv2.imread(temp_path)
                st.image(get_sleek_plot(img, st.session_state.model), channels="BGR")

# --- DATASET ANALYSIS ---
elif current_page == "Dataset Analysis":
    st.title("📁 Sleek Dataset Explorer")
    files = [f for f in os.listdir("datasets") if f.endswith(('.jpg', '.png'))]
    if files:
        sel_img = st.selectbox("Select Dataset Image", files)
        img = cv2.imread(os.path.join("datasets", sel_img))
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**🟢 Model: {st.session_state.model_name}**")
        c1.image(get_sleek_plot(img, st.session_state.model), channels="BGR")
        
        if st.session_state.secondary_model:
            c2.markdown(f"**🔵 Model: {st.session_state.secondary_name}**")
            c2.image(get_sleek_plot(img, st.session_state.secondary_model), channels="BGR")
    else: st.error("No images in /datasets")

# --- EVALUATION DASHBOARD (Fixed Indentation) ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Training Performance Dashboard")
    
    # --- SECTION 1: TOP SUMMARY (CSV Se Metrics) ---
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        df.columns = df.columns.str.strip() # Extra spaces hatane ke liye
        last_results = df.iloc[-1]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP@.5", f"{last_results.get('metrics/mAP50(B)', 0):.3f}")
        m2.metric("mAP@.5:.95", f"{last_results.get('metrics/mAP50-95(B)', 0):.3f}")
        m3.metric("Precision", f"{last_results.get('metrics/precision(B)', 0):.3f}")
        m4.metric("Recall", f"{last_results.get('metrics/recall(B)', 0):.3f}")
        st.divider()

    # --- SECTION 2: CORE PERFORMANCE ---
    col_old1, col_old2 = st.columns(2)
    with col_old1:
        st.markdown("### 📉 Training Results Matrix")
        if os.path.exists("analysis/results.png"): 
            st.image("analysis/results.png", use_container_width=True)
    with col_old2:
        st.markdown("### 🎯 Confusion Matrix")
        if os.path.exists("analysis/confusion_matrix.png"): 
            st.image("analysis/confusion_matrix.png", use_container_width=True)

    st.divider()
    st.subheader("🧪 Additional Detailed Metrics")

    # --- SECTION 3: DETAILED CURVES (Fixed Labels) ---
    col_new1, col_new2, col_new3 = st.columns(3)
    
    with col_new1:
        st.markdown("### 📉 Loss Curve")
        # Fix: Loss section mein loss_curve file
        if os.path.exists("analysis/loss_curve.png"):
            st.image("analysis/loss_curve.png", caption="Train vs Val Box/Class Loss")
        else:
            st.info("File 'loss_curve.png' not found")

    with col_new2:
        st.markdown("### 📈 mAP Curve")
        # Fix: mAP section mein map_curve file
        if os.path.exists("analysis/map_curve.png"):
            st.image("analysis/map_curve.png", caption="mAP@0.5 and mAP@0.5:0.95")
        else:
            st.info("File 'map_curve.png' not found")

    with col_new3:
        st.markdown("### 🎯 Precision-Recall")
        if os.path.exists("analysis/BoxPR_curve.png"):
            st.image("analysis/BoxPR_curve.png", caption="PR Curve over Epochs")
        elif os.path.exists("analysis/F1_curve.png"): # Fallback agar PR nahi hai
            st.image("analysis/F1_curve.png", caption="F1-Confidence Curve")
        else:
            st.info("Upload 'BoxPR_curve.png' or 'F1_curve.png'")

    # --- SECTION 4: RAW DATA ---
    with st.expander("📂 View Training Logs (CSV Data)"):
        if os.path.exists("analysis/results.csv"):
            st.dataframe(df, use_container_width=True)
# --- WEBCAM DETECTION (Optimized) ---
elif current_page == "Webcam Detection":
    # Header logic
    model_display_name = st.session_state.get('model_name', 'Model')
    st.title(f"🎥 Live Feed: {model_display_name}")
    
    # 1. RTC Configuration (STUN Servers)
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
    )
    
    # 2. Video Processor Class
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            # Session state se model uthayein
            self.model = st.session_state.get('model', None)

        def recv(self, frame):
            # Frame ko ndarray mein convert karein
            img = frame.to_ndarray(format="bgr24")
            
            # Inference logic
            if self.model is not None:
                # YOLO Inference
                results = self.model(img, conf=0.5) 
                annotated_frame = results[0].plot()
            else:
                annotated_frame = img
                
            # Wapas VideoFrame mein convert karein
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # 3. Streamer logic
    if st.session_state.get('model') is not None:
        webrtc_streamer(
            key="yolo-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.warning("⚠️ Please load a model from the 'Model Selection' page first!")

# --- MODEL COMPARISON (10 GRAPHS ADDED) ---
elif current_page == "Model Comparison":
    st.title("⚖️ Advanced Benchmarking (10-Graph Matrix)")
    
    m1 = st.session_state.model_name if st.session_state.model else "Best.pt"
    m2 = st.session_state.secondary_name if st.session_state.secondary_model else "YOLOv8n.pt"

    df = pd.DataFrame({
        "Model": [m1, m2, "Baseline"],
        "Precision": [0.88, 0.72, 0.65], "Recall": [0.84, 0.70, 0.60],
        "mAP50": [0.91, 0.75, 0.68], "Latency_ms": [15, 8, 5],
        "F1": [0.86, 0.71, 0.62], "Params_M": [8.5, 3.2, 1.0],
        "mAP50-95": [0.65, 0.45, 0.35], "Throughput": [65, 120, 200]
    })

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Precision Comparison (Bar)")
        st.plotly_chart(px.bar(df, x="Model", y="Precision", color="Model"), use_container_width=True)
        st.subheader("3. Latency Trend (Line)")
        st.plotly_chart(px.line(df, x="Model", y="Latency_ms", markers=True), use_container_width=True)
        st.subheader("5. mAP50 Confidence (Area)")
        st.plotly_chart(px.area(df, x="Model", y="mAP50"), use_container_width=True)
        st.subheader("7. Efficiency Funnel")
        st.plotly_chart(px.funnel(dict(number=[100, 80, 60, 40], stage=["Input", "Boxes", "Conf", "Final"]), x='number', y='stage'), use_container_width=True)
        st.subheader("9. Parameter Count (Scatter)")
        st.plotly_chart(px.scatter(df, x="Params_M", y="mAP50", size="Latency_ms", color="Model"), use_container_width=True)

    with c2:
        st.subheader("2. Recall Distribution (Pie)")
        st.plotly_chart(px.pie(df, names="Model", values="Recall", hole=0.3), use_container_width=True)
        st.subheader("4. Metrics Correlation (Heatmap)")
        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True), use_container_width=True)
        st.subheader("6. Performance Waterfall")
        st.plotly_chart(go.Figure(go.Waterfall(x=df["Model"], y=[0.68, 0.07, 0.16], measure=["relative"]*3)), use_container_width=True)
        st.subheader("8. F1 Score (Radar-style Bar)")
        st.plotly_chart(px.bar(df, x="Model", y="F1", color="F1"), use_container_width=True)
        st.subheader("10. Throughput vs mAP (Bubble)")
        st.plotly_chart(px.scatter(df, x="Throughput", y="mAP50-95", size="Params_M", color="Model"), use_container_width=True)
