import streamlit as st
import os
import cv2
import time
import tempfile
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="YOLOv8 Enterprise System", layout="wide")

# ======================================================
# CUSTOM CSS (UI Friendly)
# ======================================================
st.markdown("""
<style>
.sidebar .sidebar-content {
    padding-top: 40px;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# GLOBAL PATHS
# ======================================================
OUTPUT_DIR = "outputs"
DATASET_DIR = "dataset"
FAILURE_DIR = os.path.join(OUTPUT_DIR, "failures")
TRAIN_RUN = "runs/detect/train"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FAILURE_DIR, exist_ok=True)

# ======================================================
# MODEL LOADING
# ======================================================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📌 Navigation Panel")
page = st.sidebar.radio("Select Module", [
    "Image Detection",
    "Video Detection",
    "Live Webcam",
    "Evaluation",
    "Failure Cases",
    "Model Comparison"
])

model_path = st.sidebar.text_input("Model Path", "yolov8n.pt")
model = load_model(model_path)

# ======================================================
# IMAGE DETECTION FUNCTION
# ======================================================
def image_detection():
    st.title("🖼 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if uploaded:
        image = Image.open(uploaded)
        results = model(image)

        annotated = results[0].plot()

        save_path = os.path.join(OUTPUT_DIR, f"image_{int(time.time())}.jpg")
        cv2.imwrite(save_path, annotated)

        st.image(annotated, channels="BGR", use_column_width=True)

        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Output", f, file_name="detected_image.jpg")

    # Dataset Compare
    st.subheader("📂 Compare with Dataset")
    dataset_images = os.listdir(DATASET_DIR)

    if dataset_images:
        selected = st.selectbox("Select Dataset Image", dataset_images)
        if selected:
            img = Image.open(os.path.join(DATASET_DIR, selected))
            st.image(img, width=400)
    else:
        st.info("Dataset folder empty")

# ======================================================
# VIDEO DETECTION FUNCTION
# ======================================================
def video_detection():
    st.title("🎥 Video Detection")

    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        save_path = os.path.join(OUTPUT_DIR, f"video_{int(time.time())}.mp4")
        out = cv2.VideoWriter(save_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps,
                              (width, height))

        frame_display = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            out.write(annotated)
            frame_display.image(annotated, channels="BGR")

            count += 1
            progress.progress(min(count/total_frames, 1.0))

        cap.release()
        out.release()

        st.success("Video Processed Successfully")

        with open(save_path, "rb") as f:
            st.download_button("⬇ Download Processed Video", f)

# ======================================================
# WEBCAM FUNCTION
# ======================================================
def webcam_detection():
    st.title("📷 Live Webcam Detection")

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        frame_display = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break

            results = model(frame)
            annotated = results[0].plot()

            save_path = os.path.join(OUTPUT_DIR, f"webcam_{int(time.time())}.jpg")
            cv2.imwrite(save_path, annotated)

            frame_display.image(annotated, channels="BGR")

            if stop:
                break

        cap.release()

# ======================================================
# EVALUATION FUNCTION
# ======================================================
def evaluation():
    st.title("📊 Model Evaluation")

    results_csv = os.path.join(TRAIN_RUN, "results.csv")

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)

        st.subheader("Loss Curve")
        st.line_chart(df[['train/box_loss','val/box_loss']])

        st.subheader("mAP Scores")
        st.line_chart(df[['metrics/mAP50(B)','metrics/mAP50-95(B)']])

        st.subheader("Precision & Recall")
        st.line_chart(df[['metrics/precision(B)','metrics/recall(B)']])

        cm_path = os.path.join(TRAIN_RUN, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.subheader("Confusion Matrix")
            st.image(cm_path)
    else:
        st.warning("No training results found")

# ======================================================
# FAILURE CASES FUNCTION
# ======================================================
def failure_cases():
    st.title("❌ Failure Cases")

    failures = os.listdir(FAILURE_DIR)

    if failures:
        for img in failures:
            st.image(os.path.join(FAILURE_DIR, img), width=400)
    else:
        st.info("No failure cases saved yet")

# ======================================================
# MODEL COMPARISON FUNCTION
# ======================================================
def model_comparison():
    st.title("📈 Advanced Model Comparison")

    data = {
        "Model":["YOLOv8n","YOLOv8s","YOLOv8m"],
        "mAP50":[0.55,0.62,0.68],
        "mAP50-95":[0.32,0.41,0.47],
        "Recall":[0.60,0.65,0.72],
        "Precision":[0.58,0.67,0.74]
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.plotly_chart(px.line(df, x="Model", y="mAP50"))
    st.plotly_chart(px.area(df, x="Model", y="Recall"))
    st.plotly_chart(px.bar(df, x="Model", y="Precision"))
    st.plotly_chart(px.pie(df, names="Model", values="mAP50"))
    st.plotly_chart(px.funnel(df, x="mAP50", y="Model"))

    fig = go.Figure(go.Waterfall(x=df["Model"], y=df["mAP50"]))
    st.plotly_chart(fig)

    csv = df.to_csv(index=False)
    st.download_button("⬇ Download Comparison Report", csv, file_name="comparison.csv")

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
