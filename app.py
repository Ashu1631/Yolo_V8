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
if "fps_history" not in st.session_state:
    st.session_state.fps_history = {}

# ================= 2. LOGIN SYSTEM =================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;color:#00ffff'>🚀 Ashu YOLO Enterprise</h1>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🔐 Login")
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                # Simple check - you can link your users.yaml here
                if user == "admin" and pw == "admin123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
    st.stop()

# ================= 3. NAVIGATION =================
pages = ["Model Selection", "Upload & Detect", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
# Sidebar navigation syncing with session_state for auto-move
query_params = st.query_params
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

# ================= 5. PAGE LOGIC =================

# --- MODEL SELECTION ---
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
            # REQ: Automatic move to "Upload & Detect"
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --- UPLOAD & DETECT ---
elif current_page == "Upload & Detect":
    if not st.session_state.model:
        st.warning("⚠️ Please select a model first from the Selection page.")
        st.stop()
    
    st.title(f"🔍 Detection Engine ({st.session_state.model_name})")
    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded:
        file_path = os.path.join("outputs", uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if uploaded.name.endswith(".mp4"):
            # VIDEO PROCESSING
            cap = cv2.VideoCapture(file_path)
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                t_start = time.time()
                results = st.session_state.model(frame)
                fps = 1 / (time.time() - t_start)
                
                # REQ: FPS History for Graphs
                st.session_state.fps_history[st.session_state.model_name].append(fps)
                
                res_frame = results[0].plot()
                frame_placeholder.image(res_frame, channels="BGR")
            cap.release()
            
            # REQ: Performance graph only for Video
            st.subheader("📈 Performance Metrics")
            fps_df = pd.DataFrame(st.session_state.fps_history[st.session_state.model_name], columns=["FPS"])
            st.plotly_chart(px.line(fps_df, title="Inference Speed (FPS)"), use_container_width=True)

        else:
            # IMAGE PROCESSING
            img = cv2.imread(file_path)
            results = st.session_state.model(img)
            res_img = results[0].plot()
            st.image(res_img, channels="BGR", caption="Processed Image")
            
            # REQ: Auto Save
            save_path = save_result(res_img, "images")
            st.success(f"Result auto-saved to {save_path}")
            
            # Failure handling
            extract_failures(results, img, uploaded.name)

# --- EVALUATION DASHBOARD ---
elif current_page == "Evaluation Dashboard":
    st.title("📊 Model Training Evaluation")
    st.info("Showing YOLOv8 Stack Overflow style loss plots.")
    
    # REQ: Show Box Loss (Train vs Val)
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(df, x=df.index, y=['train/box_loss', 'val/box_loss'], title="Box Loss Trend"))
        with col2:
            st.plotly_chart(px.line(df, x=df.index, y=['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], title="mAP Precision"))
    else:
        # Placeholder for visual requirement
        st.warning("Upload 'results.csv' to /analysis folder to see real-time plots.")
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/results_plots.png")

# --- MODEL COMPARISON ---
elif current_page == "Model Comparison":
    st.title("⚖️ Competitive Model Benchmarking")
    
    # Mock Data for various charts
    comp_data = {
        "Model": ["YOLOv8n", "YOLOv8s", "Your Custom Model"],
        "mAP50": [0.74, 0.81, 0.88],
        "Latency(ms)": [8, 12, 15],
        "Accuracy": [70, 78, 85]
    }
    df = pd.DataFrame(comp_data)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df, x="Model", y="mAP50", color="Model", title="Bar Chart: Precision"))
        st.plotly_chart(px.line(df, x="Model", y="Latency(ms)", title="Line Chart: Latency Trend"))
    with c2:
        st.plotly_chart(px.pie(df, names="Model", values="Accuracy", title="Pie Chart: Accuracy Distribution"))
        st.plotly_chart(px.area(df, x="Model", y="mAP50", title="Area Chart: Performance Coverage"))

    st.subheader("Heatmap & Data Table")
    st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=True, title="Feature Heatmap"))
    
    # REQ: Download Report
    st.download_button("📥 Download Full Report", df.to_csv().encode('utf-8'), "report.csv", "text/csv")

# --- WEBCAM DETECTION ---
elif current_page == "Webcam Detection":
    st.title("🎥 Real-time Webcam Feed")
    if not st.session_state.model:
        st.error("Please load a model first!")
    else:
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = st.session_state.model(img)
                return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

        webrtc_streamer(key="yolo-webcam", video_processor_factory=VideoProcessor)

# --- FAILURE CASES ---
elif current_page == "Failure Cases":
    st.title("⚠️ Automated Failure Analysis")
    fails = os.listdir("failure_cases")
    if fails:
        selected_fail = st.selectbox("Review Failures", fails)
        st.image(os.path.join("failure_cases", selected_fail))
    else:
        st.success("No critical failure cases detected so far!")
