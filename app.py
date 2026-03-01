import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

st.set_page_config(page_title="YOLOv8 Pro Dashboard", layout="wide")
st.title("🚀 YOLOv8 Professional Detection Dashboard")

# ---------------------------
# Sidebar Controls
# ---------------------------

st.sidebar.header("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "best.pt"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.25
)

mode = st.sidebar.radio(
    "Select Mode",
    ["Single Image", "Multiple Images", "Video"]
)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_option)

# ---------------------------
# Helper Function
# ---------------------------

def process_result(result):
    if result.boxes:
        names = result.names
        data = []

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= confidence_threshold:
                cls = int(box.cls[0])
                data.append({
                    "Object": names[cls],
                    "Confidence": round(conf, 2)
                })

        return pd.DataFrame(data)
    return pd.DataFrame()

# ---------------------------
# SINGLE IMAGE MODE
# ---------------------------

if mode == "Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = model(image)
        result = results[0]

        annotated = result.plot()
        st.image(annotated, caption="Detected Image", use_column_width=True)

        df = process_result(result)

        if not df.empty:
            st.subheader("📊 Detection Table")
            st.dataframe(df)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Object Count")
                count_df = df["Object"].value_counts()
                fig1, ax1 = plt.subplots()
                count_df.plot(kind="bar", ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.subheader("📉 Confidence Scores")
                fig2, ax2 = plt.subplots()
                ax2.bar(df["Object"], df["Confidence"])
                ax2.set_ylim(0,1)
                st.pyplot(fig2)

        else:
            st.error("No objects detected.")
            st.markdown("### Possible Failure Causes:")
            st.markdown("""
            - Low image quality  
            - Object too small  
            - Object not in trained dataset  
            - Poor lighting  
            - Wrong model selected  
            """)

# ---------------------------
# MULTIPLE IMAGE MODE
# ---------------------------

elif mode == "Multiple Images":
    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg","png","jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_data = []

        for file in uploaded_files:
            image = Image.open(file)
            results = model(image)
            result = results[0]
            df = process_result(result)

            if not df.empty:
                df["Image"] = file.name
                all_data.append(df)

        if all_data:
            final_df = pd.concat(all_data)
            st.subheader("📊 Batch Detection Results")
            st.dataframe(final_df)
        else:
            st.warning("No detections in uploaded images.")

# ---------------------------
# VIDEO MODE
# ---------------------------

elif mode == "Video":
    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR")

        cap.release()

# ---------------------------
# mAP & Confusion Matrix
# ---------------------------

st.sidebar.header("📊 Model Metrics")

if st.sidebar.button("Show mAP Metrics (if available)"):
    if os.path.exists("runs/detect/train/results.csv"):
        metrics = pd.read_csv("runs/detect/train/results.csv")
        st.subheader("📈 mAP Metrics")
        st.line_chart(metrics[["metrics/mAP50(B)", "metrics/mAP50-95(B)"]])
    else:
        st.warning("Training results not found.")

if st.sidebar.button("Show Confusion Matrix (if available)"):
    cm_path = "runs/detect/train/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.subheader("📉 Confusion Matrix")
        st.image(cm_path)
    else:
        st.warning("Confusion matrix not found.")
