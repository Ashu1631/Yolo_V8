import streamlit as st
import os
import cv2
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image
import time
import shutil

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="YOLOv8 Detection System", layout="wide")

# Custom CSS for spacing
st.markdown("""
<style>
section[data-testid="stSidebar"] .css-ng1t4o {
    padding-top: 20px;
}
.sidebar .sidebar-content {
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR = "outputs"
DATASET_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Image Detection",
    "Video Detection",
    "Webcam",
    "Evaluation",
    "Failure Cases",
    "Model Comparison"
])

model_path = st.sidebar.text_input("Model Path", "yolov8n.pt")
model = YOLO(model_path)

# =========================
# IMAGE DETECTION
# =========================
if page == "Image Detection":
    st.title("Image Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        results = model(image)

        result_img = results[0].plot()
        save_path = os.path.join(OUTPUT_DIR, f"img_{int(time.time())}.jpg")
        cv2.imwrite(save_path, result_img)

        st.image(result_img, caption="Detected Image", use_column_width=True)

        with open(save_path, "rb") as f:
            st.download_button("Download Output", f, file_name="detected.jpg")

    # Dataset compare option
    st.subheader("Compare with Dataset Image")
    dataset_images = os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else []
    selected = st.selectbox("Select Dataset Image", dataset_images)

    if selected:
        dataset_img = Image.open(os.path.join(DATASET_DIR, selected))
        st.image(dataset_img, caption="Dataset Image", width=300)

# =========================
# VIDEO DETECTION
# =========================
elif page == "Video Detection":
    st.title("Video Detection")

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
                              fps, (width, height))

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
        out.release()

        st.success("Video Processed Successfully")

        with open(save_path, "rb") as f:
            st.download_button("Download Processed Video", f)

# =========================
# WEBCAM
# =========================
elif page == "Webcam":
    st.title("Live Webcam Detection")

    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam not accessible")
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            save_path = os.path.join(OUTPUT_DIR, f"webcam_{int(time.time())}.jpg")
            cv2.imwrite(save_path, annotated_frame)

            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()

# =========================
# EVALUATION
# =========================
elif page == "Evaluation":
    st.title("Model Evaluation")

    results_csv = "runs/detect/train/results.csv"

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)

        st.line_chart(df[['train/box_loss','val/box_loss']])
        st.line_chart(df[['metrics/mAP50(B)','metrics/mAP50-95(B)']])
        st.line_chart(df[['metrics/precision(B)','metrics/recall(B)']])

        # Confusion matrix (if exists)
        cm_path = "runs/detect/train/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")
    else:
        st.warning("Train model first")

# =========================
# FAILURE CASES
# =========================
elif page == "Failure Cases":
    st.title("Failure Cases")

    failure_dir = os.path.join(OUTPUT_DIR, "failures")
    os.makedirs(failure_dir, exist_ok=True)

    failures = os.listdir(failure_dir)
    if failures:
        for f in failures:
            st.image(os.path.join(failure_dir, f), width=300)
    else:
        st.info("No failure cases found")

# =========================
# MODEL COMPARISON
# =========================
elif page == "Model Comparison":
    st.title("Model Comparison Dashboard")

    sample_data = {
        "Model":["YOLOv8n","YOLOv8s","YOLOv8m"],
        "mAP50":[0.55,0.62,0.68],
        "mAP50-95":[0.32,0.41,0.47],
        "Recall":[0.60,0.65,0.72],
        "Precision":[0.58,0.67,0.74]
    }

    df = pd.DataFrame(sample_data)

    st.subheader("Comparison Table")
    st.dataframe(df)

    fig_line = px.line(df, x="Model", y="mAP50")
    st.plotly_chart(fig_line)

    fig_area = px.area(df, x="Model", y="Recall")
    st.plotly_chart(fig_area)

    fig_bar = px.bar(df, x="Model", y="Precision")
    st.plotly_chart(fig_bar)

    fig_pie = px.pie(df, names="Model", values="mAP50")
    st.plotly_chart(fig_pie)

    fig_funnel = px.funnel(df, x="mAP50", y="Model")
    st.plotly_chart(fig_funnel)

    fig_waterfall = go.Figure(go.Waterfall(
        x=df["Model"],
        y=df["mAP50"],
    ))
    st.plotly_chart(fig_waterfall)

    csv = df.to_csv(index=False)
    st.download_button("Download Report", csv, file_name="model_comparison.csv")
