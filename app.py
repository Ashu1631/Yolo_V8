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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# =====================================================
# GLOBAL DIRECTORIES
# =====================================================
OUTPUT_DIR = "outputs"
DATASET_DIR = "dataset"
FAILURE_DIR = os.path.join(OUTPUT_DIR, "failures")
RUNS_DIR = "runs/detect"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)

# =====================================================
# CUSTOM SIDEBAR STYLE (LIKE IMAGE)
# =====================================================
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #1f2937;
    padding-top: 20px;
}
.nav-title {
    font-size: 22px;
    font-weight: 600;
    color: white;
    margin-bottom: 20px;
}
.stButton > button {
    width: 100%;
    border-radius: 8px;
    padding: 10px;
    font-weight: 500;
    margin-bottom: 10px;
    border: none;
}
.stButton > button:hover {
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown('<div class="nav-title">🚀 Navigation</div>', unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

with st.sidebar:
    if st.button("👉 Model Selection"):
        st.session_state.page = "Model Selection"
    if st.button("Upload & Detect"):
        st.session_state.page = "Image Detection"
    if st.button("Video Detection"):
        st.session_state.page = "Video Detection"
    if st.button("Webcam Detection"):
        st.session_state.page = "Live Webcam"
    if st.button("Evaluation Dashboard"):
        st.session_state.page = "Evaluation"
    if st.button("Failure Cases"):
        st.session_state.page = "Failure Cases"
    if st.button("Model Comparison"):
        st.session_state.page = "Model Comparison"

page = st.session_state.page

# =====================================================
# MODEL SELECTION
# =====================================================
default_models = ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt","yolov8x.pt"]

trained_models = []
if os.path.exists(RUNS_DIR):
    for root, dirs, files in os.walk(RUNS_DIR):
        for f in files:
            if f.endswith(".pt"):
                trained_models.append(os.path.join(root, f))

model_list = default_models + trained_models

@st.cache_resource
def load_model(path):
    return YOLO(path)

if page == "Model Selection":
    st.title("📦 Model Selection")

    selected_model = st.selectbox("Select Model", model_list)
    uploaded_model = st.file_uploader("Or Upload Custom Model (.pt)", type=["pt"])

    if uploaded_model:
        temp_model = "temp_uploaded_model.pt"
        with open(temp_model, "wb") as f:
            f.write(uploaded_model.read())
        model_path = temp_model
    else:
        model_path = selected_model

    model = load_model(model_path)
    st.success(f"Loaded Model: {os.path.basename(model_path)}")

    st.session_state.model_path = model_path

# Load model globally if already selected
if "model_path" in st.session_state:
    model = load_model(st.session_state.model_path)
else:
    model = load_model("yolov8n.pt")

confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)
iou_thresh = st.sidebar.slider("IoU", 0.0, 1.0, 0.45)

# =====================================================
# IMAGE DETECTION
# =====================================================
if page == "Image Detection":
    st.title("🖼 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        results = model(image, conf=confidence, iou=iou_thresh)

        annotated = results[0].plot()
        save_path = os.path.join(OUTPUT_DIR, f"img_{int(time.time())}.jpg")
        cv2.imwrite(save_path, annotated)

        st.image(annotated, channels="BGR", use_column_width=True)

        if len(results[0].boxes) == 0:
            cv2.imwrite(os.path.join(FAILURE_DIR, f"fail_{int(time.time())}.jpg"), annotated)

        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Output", f)

# =====================================================
# VIDEO DETECTION
# =====================================================
elif page == "Video Detection":
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
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        frame_display = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated = results[0].plot()

            out.write(annotated)
            frame_display.image(annotated, channels="BGR")

            count += 1
            if total_frames > 0:
                progress.progress(min(count/total_frames,1.0))

        cap.release()
        out.release()

        st.success("Video Processed Successfully")
        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Processed Video", f)

# =====================================================
# WEBCAM
# =====================================================
elif page == "Live Webcam":
    st.title("📷 Live Webcam")

    if "webcam" not in st.session_state:
        st.session_state.webcam = False

    if st.button("Start Webcam"):
        st.session_state.webcam = True
    if st.button("Stop Webcam"):
        st.session_state.webcam = False

    frame_display = st.empty()

    if st.session_state.webcam:
        cap = cv2.VideoCapture(0)

        while st.session_state.webcam:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_thresh)
            annotated = results[0].plot()

            save_path = os.path.join(OUTPUT_DIR, f"webcam_{int(time.time())}.jpg")
            cv2.imwrite(save_path, annotated)

            frame_display.image(annotated, channels="BGR")

        cap.release()

# =====================================================
# EVALUATION
# =====================================================
elif page == "Evaluation":
    st.title("📊 Evaluation Dashboard")

    runs = os.listdir(RUNS_DIR) if os.path.exists(RUNS_DIR) else []
    selected_run = st.selectbox("Select Run", runs)

    if selected_run:
        csv_path = os.path.join(RUNS_DIR, selected_run, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            st.line_chart(df.filter(regex="loss"))
            st.line_chart(df.filter(regex="mAP"))
            st.line_chart(df.filter(regex="precision|recall"))

            cm_path = os.path.join(RUNS_DIR, selected_run, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path)

# =====================================================
# FAILURE CASES
# =====================================================
elif page == "Failure Cases":
    st.title("❌ Failure Cases")

    failures = os.listdir(FAILURE_DIR)
    if failures:
        for img in failures:
            st.image(os.path.join(FAILURE_DIR, img), width=400)
    else:
        st.info("No failure cases saved.")

# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "Model Comparison":
    st.title("📈 Model Comparison")

    comparison = []

    if os.path.exists(RUNS_DIR):
        for run in os.listdir(RUNS_DIR):
            csv_path = os.path.join(RUNS_DIR, run, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                last = df.iloc[-1]

                comparison.append({
                    "Run": run,
                    "mAP50": last.filter(like="mAP50").values[0] if len(last.filter(like="mAP50"))>0 else 0,
                    "Precision": last.filter(like="precision").values[0] if len(last.filter(like="precision"))>0 else 0,
                    "Recall": last.filter(like="recall").values[0] if len(last.filter(like="recall"))>0 else 0
                })

    if comparison:
        df = pd.DataFrame(comparison)
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
        st.warning("No training runs available.")
