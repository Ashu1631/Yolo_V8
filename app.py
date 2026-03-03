import streamlit as st
import os
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="YOLOv8 Enterprise AI System", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.sidebar .sidebar-content {padding-top: 30px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# GLOBAL DIRECTORIES
# ======================================================
OUTPUT_DIR = "outputs"
DATASET_DIR = "dataset"
FAILURE_DIR = os.path.join(OUTPUT_DIR, "failures")
RUNS_DIR = "runs/detect"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)

# ======================================================
# MODEL SELECTION SYSTEM
# ======================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Module", [
    "Image Detection",
    "Video Detection",
    "Live Webcam",
    "Evaluation",
    "Failure Cases",
    "Model Comparison"
])

st.sidebar.subheader("🤖 Model Selection")

default_models = ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"]

trained_models = []
if os.path.exists(RUNS_DIR):
    for root, dirs, files in os.walk(RUNS_DIR):
        for f in files:
            if f.endswith(".pt"):
                trained_models.append(os.path.join(root, f))

model_list = default_models + trained_models
selected_model = st.sidebar.selectbox("Select Model", model_list)

uploaded_model = st.sidebar.file_uploader("Or Upload Custom Model (.pt)", type=["pt"])

if uploaded_model:
    temp_model_path = "temp_uploaded_model.pt"
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.read())
    model_path = temp_model_path
else:
    model_path = selected_model

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)
st.sidebar.success(f"Loaded: {os.path.basename(model_path)}")

# Detection Settings
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45)

# ======================================================
# IMAGE DETECTION
# ======================================================
def image_detection():
    st.title("🖼 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded:
        image = Image.open(uploaded)
        results = model(image, conf=confidence, iou=iou_thresh)

        annotated = results[0].plot()

        save_path = os.path.join(OUTPUT_DIR, f"image_{int(time.time())}.jpg")
        cv2.imwrite(save_path, annotated)

        st.image(annotated, channels="BGR", use_column_width=True)

        # Failure detection
        if len(results[0].boxes) == 0:
            fail_path = os.path.join(FAILURE_DIR, f"fail_{int(time.time())}.jpg")
            cv2.imwrite(fail_path, annotated)

        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Output", f)

# ======================================================
# VIDEO DETECTION
# ======================================================
def video_detection():
    st.title("🎥 Video Detection")

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        save_path = os.path.join(OUTPUT_DIR, f"video_{int(time.time())}.mp4")
        out = cv2.VideoWriter(save_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (width, height))

        frame_display = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated = results[0].plot()

            out.write(annotated)
            frame_display.image(annotated, channels="BGR")

            frame_count += 1
            if total_frames > 0:
                progress.progress(min(frame_count/total_frames, 1.0))

        cap.release()
        out.release()

        st.success("Video Processed Successfully")

        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Processed Video", f)

# ======================================================
# WEBCAM (STABLE VERSION)
# ======================================================
def webcam_detection():
    st.title("📷 Live Webcam Detection")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if start:
        st.session_state.webcam_running = True
    if stop:
        st.session_state.webcam_running = False

    frame_display = st.empty()

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)

        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated = results[0].plot()

            save_path = os.path.join(OUTPUT_DIR, f"webcam_{int(time.time())}.jpg")
            cv2.imwrite(save_path, annotated)

            frame_display.image(annotated, channels="BGR")

        cap.release()

# ======================================================
# EVALUATION
# ======================================================
def evaluation():
    st.title("📊 Model Evaluation")

    run_folders = []
    if os.path.exists(RUNS_DIR):
        run_folders = os.listdir(RUNS_DIR)

    selected_run = st.selectbox("Select Training Run", run_folders)

    results_csv = os.path.join(RUNS_DIR, selected_run, "results.csv")

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)

        st.line_chart(df.filter(regex="loss"))
        st.line_chart(df.filter(regex="mAP"))
        st.line_chart(df.filter(regex="precision|recall"))

        cm_path = os.path.join(RUNS_DIR, selected_run, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path)
    else:
        st.warning("No results found")

# ======================================================
# FAILURE CASES
# ======================================================
def failure_cases():
    st.title("❌ Failure Cases")

    failures = os.listdir(FAILURE_DIR)

    if failures:
        for img in failures:
            st.image(os.path.join(FAILURE_DIR, img), width=400)
    else:
        st.info("No failure cases saved")

# ======================================================
# MODEL COMPARISON (AUTO FROM RUNS)
# ======================================================
def model_comparison():
    st.title("📈 Model Comparison Dashboard")

    comparison_data = []

    if os.path.exists(RUNS_DIR):
        for run in os.listdir(RUNS_DIR):
            csv_path = os.path.join(RUNS_DIR, run, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                last = df.iloc[-1]

                comparison_data.append({
                    "Run": run,
                    "mAP50": last.filter(like="mAP50").values[0] if "mAP50" in str(last.index) else 0,
                    "Precision": last.filter(like="precision").values[0] if "precision" in str(last.index) else 0,
                    "Recall": last.filter(like="recall").values[0] if "recall" in str(last.index) else 0
                })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        st.dataframe(df)

        st.plotly_chart(px.line(df, x="Run", y="mAP50"))
        st.plotly_chart(px.area(df, x="Run", y="Recall"))
        st.plotly_chart(px.bar(df, x="Run", y="Precision"))
        st.plotly_chart(px.pie(df, names="Run", values="mAP50"))
        st.plotly_chart(px.funnel(df, x="mAP50", y="Run"))

        fig = go.Figure(go.Waterfall(x=df["Run"], y=df["mAP50"]))
        st.plotly_chart(fig)

        st.download_button("⬇ Download Comparison CSV", df.to_csv(index=False))
    else:
        st.warning("No training runs available")

# ======================================================
# ROUTER
# ======================================================
if page == "Image Detection":
    image_detection()
elif page == "Video Detection":
    video_detection()
elif page == "Live Webcam":
    webcam_detection()
elif page == "Evaluation":
    evaluation()
elif page == "Failure Cases":
    failure_cases()
elif page == "Model Comparison":
    model_comparison()
