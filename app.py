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
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis"]
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
pages = ["Model Selection", "Upload & Detect", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
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

# --- MODEL SELECTION (Auto-Nav) ---
if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model to Initialize", ["-- Select --"] + models)
    
    if selected != "-- Select --":
        with st.spinner("Initializing AI Engine..."):
            st.session_state.model = YOLO(selected)
            st.session_state.model_name = selected
            st.session_state.fps_history[selected] = []
            st.success(f"Model {selected} Loaded!")
            time.sleep(1)
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --- UPLOAD & DETECT (Auto Save + Video FPS) ---
elif current_page == "Upload & Detect":
    st.title(f"🔍 Detection Engine ({st.session_state.model_name})")
    if not st.session_state.model:
        st.warning("⚠️ Please select a model first from the Selection page.")
        st.stop()
    
    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded:
        file_path = os.path.join("outputs", uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if uploaded.name.endswith(".mp4"):
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
            st.image(res_img, channels="BGR", caption="Processed Image")
            save_path = save_result(res_img, "images")
            st.success(f"Result auto-saved to {save_path}")
            extract_failures(results, img, uploaded.name)

# --- EVALUATION DASHBOARD (Loss Plots) ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Training Evaluation")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.line(df, y=['train/box_loss', 'val/box_loss'], title="Box Loss Trend"))
        with col2: st.plotly_chart(px.line(df, y=['metrics/mAP50(B)'], title="mAP Precision"))
    else:
        st.info("Showing Reference YOLOv8 Metrics. (Upload 'results.csv' to /analysis/ for live data)")
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/results_plots.png")

# --- MODEL COMPARISON (7 Graphs + Report) ---
elif current_page == "Model Comparison":
    st.title("⚖️ Competitive Model Benchmarking")
    data = {
        "Model": ["YOLOv8n", "YOLOv8s", "Your_Custom_Model"],
        "mAP50": [0.74, 0.81, 0.88],
        "Latency_ms": [8, 12, 18],
        "Precision": [0.70, 0.77, 0.85],
        "Recall": [0.68, 0.75, 0.82],
        "F1_Score": [0.71, 0.76, 0.84]
    }
    df = pd.DataFrame(data)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df, x="Model", y="mAP50", color="Model", title="1. Bar Chart: mAP50"))
        st.plotly_chart(px.area(df, x="Model", y="Precision", title="3. Area Chart: Precision Coverage"))
        # 5. Waterfall Chart
        fig_water = go.Figure(go.Waterfall(x=df["Model"], y=[0.74, 0.07, 0.07], measure=["relative"]*3))
        fig_water.update_layout(title="5. Waterfall: Incremental mAP Gain")
        st.plotly_chart(fig_water)
    with c2:
        st.plotly_chart(px.line(df, x="Model", y="Latency_ms", title="2. Line Chart: Latency Trend"))
        st.plotly_chart(px.pie(df, names="Model", values="F1_Score", title="4. Pie Chart: F1 Distribution"))
        # 6. Heatmap
        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True, title="6. Heatmap: Metric Correlation"))

    st.subheader("7. Pie Table (Data Metrics)")
    st.table(df)
    st.download_button("📥 Download Full Report", df.to_csv().encode('utf-8'), "benchmark_report.csv", "text/csv")

# --- WEBCAM DETECTION ---
elif current_page == "Webcam Detection":
    st.title("🎥 Real-time Webcam Feed")
    if st.session_state.model:
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = st.session_state.model(img)
                return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")
        webrtc_streamer(key="yolo-webcam", video_processor_factory=VideoProcessor)
    else: st.error("Select model first!")

# --- FAILURE CASES ---
elif current_page == "Failure Cases":
    st.title("⚠️ Automated Failure Logs")
    fails = [f for f in os.listdir("failure_cases") if f.endswith(('.jpg', '.png'))]
    if fails:
        selected_fail = st.selectbox("Review Failures", fails)
        st.image(os.path.join("failure_cases", selected_fail))

# --- MODEL COMPARISON DASHBOARD ---
elif current_page == "Model Comparison":
    st.title("⚖️ Advanced Model Benchmarking")
    st.markdown("Is section mein aap different models ki performance compare kar sakte hain.")

    # Mock Data for Comparison (YOLOv8 variants vs Your Model)
    data = {
        "Model": ["YOLOv8n", "YOLOv8s", "Your_Custom_Model"],
        "mAP50": [0.72, 0.78, 0.88],
        "Recall": [0.68, 0.75, 0.82],
        "Precision": [0.70, 0.77, 0.85],
        "Latency_ms": [8, 12, 18],
        "F1_Score": [0.71, 0.76, 0.84],
        "CPU_Usage": [20, 35, 45]
    }
    df = pd.DataFrame(data)

    # Creating Tabs for organized view
    tab_charts, tab_stats = st.tabs(["📊 Performance Visualization", "📋 Detailed Reports"])

    with tab_charts:
        # 1. Bar Chart & 2. Line Chart
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Precision vs Model (Bar)")
            st.plotly_chart(px.bar(df, x="Model", y="Precision", color="Model", template="plotly_dark"), use_container_width=True)
        
        with col2:
            st.subheader("2. Latency Trend (Line)")
            st.plotly_chart(px.line(df, x="Model", y="Latency_ms", markers=True, title="Inference Speed (Lower is better)"), use_container_width=True)

        # 3. Area Chart & 4. Pie Chart
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. mAP50 Confidence (Area)")
            st.plotly_chart(px.area(df, x="Model", y="mAP50", title="Mean Average Precision Coverage"), use_container_width=True)
        
        with col4:
            st.subheader("4. F1 Score Distribution (Pie)")
            st.plotly_chart(px.pie(df, names="Model", values="F1_Score", hole=0.3), use_container_width=True)

        st.divider()

        # 5. Water Flow (Waterfall) Chart
        st.subheader("5. Performance Gain (Waterfall)")
        # Calculating relative gain from baseline
        fig_water = go.Figure(go.Waterfall(
            name = "Gain", orientation = "v",
            x = df["Model"],
            textposition = "outside",
            text = df["mAP50"],
            y = [0.72, 0.06, 0.10], # Relative improvements
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title = "mAP50 Incremental Improvement")
        st.plotly_chart(fig_water, use_container_width=True)

        # 6. Heatmap Correlation
        st.subheader("6. Metrics Correlation (Heatmap)")
        st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

    with tab_stats:
        # 7. Pie Table (Data Grid)
        st.subheader("7. Detailed Metrics Table")
        st.dataframe(df.style.highlight_max(axis=0, color='green'), use_container_width=True)

        # Download Report Option
        st.divider()
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Benchmark Report as CSV",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Click here to download the comparison data for your presentation."
        )
    else: st.success("No critical failure cases detected!")
