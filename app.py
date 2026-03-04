import streamlit as st
import os
import cv2
import time
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Ashu YOLO Enterprise Dashboard", layout="wide", initial_sidebar_state="expanded")

# ================= 1. DIRECTORY & SESSION SETUP =================
# Added "dataset" directory to the list
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis", "dataset"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model" not in st.session_state:
    st.session_state.model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
if "fps_history" not in st.session_state:
    st.session_state.fps_history = {}
# For side-by-side comparison
if "secondary_model" not in st.session_state:
    st.session_state.secondary_model = None

# ================= 2. LOGIN SYSTEM =================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;color:#00ffff'>🚀 Ashu YOLO Enterprise</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Login")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if user == "admin" and pw == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")
    st.stop()

# ================= 3. NAVIGATION LOGIC =================
# Added "Dataset Analysis" to the pages
pages = ["Model Selection", "Upload & Detect", "Dataset Analysis", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
current_page = st.sidebar.radio("🚀 Navigation", pages, index=pages.index(st.session_state.page))

# ================= 4. HELPER FUNCTIONS =================
def save_result(frame, folder="images"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/{folder}/det_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path

def extract_failures(results, image, filename):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0 or any(boxes.conf.cpu().numpy() < 0.25):
        cv2.imwrite(f"failure_cases/fail_{filename}", image)

# ================= 5. PAGE CONTENT =================

# --- MODEL SELECTION ---
if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Primary Model", ["-- Select --"] + models)
    secondary = st.selectbox("Select Secondary Model (For Side-by-Side Comparison)", ["None"] + models)
    
    if selected != "-- Select --":
        if st.button("Initialize Engines"):
            with st.spinner("Initializing AI Engines..."):
                st.session_state.model = YOLO(selected)
                st.session_state.model_name = selected
                st.session_state.fps_history[selected] = []
                if secondary != "None":
                    st.session_state.secondary_model = YOLO(secondary)
                    st.session_state.secondary_name = secondary
                st.success(f"Models Loaded!")
                time.sleep(1)
                st.session_state.page = "Upload & Detect"
                st.rerun()

# --- UPLOAD & DETECT (With Side-by-Side Comparison) ---
elif current_page == "Upload & Detect":
    st.title(f"🔍 Detection Engine")
    if not st.session_state.model:
        st.warning("⚠️ Please select a model first.")
        st.stop()
    
    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded:
        file_path = os.path.join("outputs", uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Logic for Side-by-Side Mode
        if st.session_state.secondary_model and uploaded.name.lower().endswith(('.jpg', '.png', '.jpeg')):
            st.subheader("⚖️ Side-by-Side Model Comparison")
            col1, col2 = st.columns(2)
            img = cv2.imread(file_path)
            
            with col1:
                st.info(f"Primary: {st.session_state.model_name}")
                res1 = st.session_state.model(img)
                st.image(res1[0].plot(), channels="BGR")
            
            with col2:
                st.info(f"Secondary: {st.session_state.secondary_name}")
                res2 = st.session_state.secondary_model(img)
                st.image(res2[0].plot(), channels="BGR")
        
        # Standard Single Inference (Images/Video)
        elif uploaded.name.endswith(".mp4"):
            cap = cv2.VideoCapture(file_path)
            st_frame = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                t_start = time.time()
                results = st.session_state.model(frame)
                st.session_state.fps_history[st.session_state.model_name].append(1 / (time.time() - t_start))
                st_frame.image(results[0].plot(), channels="BGR")
            cap.release()
            
            st.subheader("📈 Performance Metrics")
            fps_df = pd.DataFrame(st.session_state.fps_history[st.session_state.model_name], columns=["FPS"])
            st.plotly_chart(px.line(fps_df, title="Inference Speed (FPS)"), use_container_width=True)
        else:
            img = cv2.imread(file_path)
            results = st.session_state.model(img)
            res_img = results[0].plot()
            st.image(res_img, channels="BGR", caption=f"Result: {st.session_state.model_name}")
            save_path = save_result(res_img, "images")
            st.success(f"Result auto-saved to {save_path}")
            extract_failures(results, img, uploaded.name)

# --- DATASET ANALYSIS (New Sidebar Option) ---
elif current_page == "Dataset Analysis":
    st.title("📁 Dataset Inventory & Stats")
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(('.jpg', '.png'))]
    
    if dataset_files:
        st.write(f"Total Images in Dataset: **{len(dataset_files)}**")
        # Mock class distribution for comparison
        dist_data = {"Class": ["Person", "Car", "Bike", "Dog"], "Count": [120, 85, 40, 25]}
        df_dist = pd.DataFrame(dist_data)
        st.plotly_chart(px.bar(df_dist, x="Class", y="Count", color="Class", title="Class Distribution in Dataset"))
    else:
        st.info("The `/dataset` folder is currently empty. Upload your dataset images there.")

# --- EVALUATION DASHBOARD (YOLO Style Matrix) ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Training Evaluation Matrix")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        # Creating a 2x2 grid similar to YOLO training logs
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(df, y=['train/box_loss', 'val/box_loss'], title="Box Loss (Train vs Val)"))
            st.plotly_chart(px.line(df, y='metrics/mAP50(B)', title="mAP @ 50"))
        with c2:
            st.plotly_chart(px.line(df, y='metrics/recall(B)', title="Recall Curve"))
            st.plotly_chart(px.line(df, y='metrics/mAP50-95(B)', title="mAP @ 50-95"))
    else:
        st.info("Showing Official YOLOv8 Training Results Template.")
        
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/results_plots.png")

# --- MODEL COMPARISON (With Funnel Chart) ---
elif current_page == "Model Comparison":
    st.title("⚖️ Advanced Model Benchmarking")
    
    data = {
        "Model": ["YOLOv8n", "YOLOv8s", "Your_Custom_Model"],
        "mAP50": [0.72, 0.78, 0.88],
        "Recall": [0.68, 0.75, 0.82],
        "Precision": [0.70, 0.77, 0.85],
        "Latency_ms": [8, 12, 18],
        "F1_Score": [0.71, 0.76, 0.84]
    }
    df = pd.DataFrame(data)

    tab_charts, tab_stats = st.tabs(["📊 Performance Visualization", "📋 Detailed Reports"])

    with tab_charts:
        # Added Funnel Chart here
        st.subheader("0. Detection Pipeline Funnel")
        funnel_data = dict(
            number=[100, 80, 60, 55],
            stage=["Input Images", "Candidate Boxes", "Confidence Threshold", "NMS Filtered"]
        )
        st.plotly_chart(px.funnel(funnel_data, x='number', y='stage', title="Detection Flow Efficiency"))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Precision vs Model (Bar)")
            st.plotly_chart(px.bar(df, x="Model", y="Precision", color="Model"), use_container_width=True)
        with col2:
            st.subheader("2. Latency Trend (Line)")
            st.plotly_chart(px.line(df, x="Model", y="Latency_ms", markers=True), use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. mAP50 Confidence (Area)")
            st.plotly_chart(px.area(df, x="Model", y="mAP50"), use_container_width=True)
        with col4:
            st.subheader("4. F1 Score Distribution (Pie)")
            st.plotly_chart(px.pie(df, names="Model", values="F1_Score", hole=0.3), use_container_width=True)

        st.divider()
        st.subheader("5. Performance Gain (Waterfall)")
        fig_water = go.Figure(go.Waterfall(x = df["Model"], y = [0.72, 0.06, 0.10], measure = ["relative", "relative", "relative"]))
        st.plotly_chart(fig_water, use_container_width=True)

        st.subheader("6. Metrics Correlation (Heatmap)")
        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True), use_container_width=True)

    with tab_stats:
        st.subheader("7. Detailed Metrics Table")
        st.dataframe(df.style.highlight_max(axis=0, color='green'), use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Benchmark Report", data=csv, file_name="benchmark_report.csv", mime="text/csv")

# --- WEBCAM ---
elif current_page == "Webcam Detection":
    st.title("🎥 Live Stream")
    if st.session_state.model:
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                res = st.session_state.model(img)
                return av.VideoFrame.from_ndarray(res[0].plot(), format="bgr24")
        webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor)
    else:
        st.error("Select model first!")

# --- FAILURE CASES ---
elif current_page == "Failure Cases":
    st.title("⚠️ Automated Failure Logs")
    if os.path.exists("failure_cases"):
        fails = [f for f in os.listdir("failure_cases") if f.endswith(('.jpg', '.png'))]
        if fails:
            selected_fail = st.selectbox("Review Failures", fails)
            st.image(os.path.join("failure_cases", selected_fail))
        else:
            st.success("No critical failure cases detected!")
