import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import time
import torch
from PIL import Image
import tempfile

st.set_page_config(page_title="Enterprise YOLOv8 AI Platform", layout="wide")

st.sidebar.title("🚀 YOLOv8 Enterprise Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Model Selection",
        "Dataset Analytics",
        "Upload & Detect",
        "Real-time Camera",
        "Training",
        "Auto Hyperparameter Tuning",
        "Evaluation Dashboard",
        "Grad-CAM Explainability"
    ]
)

if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# 1️⃣ MODEL SELECTION
# ==========================================================
if menu == "Model Selection":

    st.header("📦 Load YOLOv8 Model")

    model_choice = st.selectbox(
        "Select Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "best.pt"]
    )

    if st.button("Load Model"):
        st.session_state.model = YOLO(model_choice)
        st.success("Model Loaded Successfully!")

# ==========================================================
# 2️⃣ DATASET ANALYTICS
# ==========================================================
elif menu == "Dataset Analytics":

    st.header("📊 Dataset Analytics")

    label_folder = "data/labels"

    if os.path.exists(label_folder):

        class_counts = {}

        for file in os.listdir(label_folder):
            with open(os.path.join(label_folder, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    cls = line.split()[0]
                    class_counts[cls] = class_counts.get(cls, 0) + 1

        df = pd.DataFrame({
            "Class": list(class_counts.keys()),
            "Count": list(class_counts.values())
        })

        st.plotly_chart(px.pie(df, names="Class", values="Count",
                               title="Class Distribution"))

        st.plotly_chart(px.bar(df, x="Class", y="Count",
                               title="Class Imbalance"))

    else:
        st.warning("Label folder not found.")

# ==========================================================
# 3️⃣ IMAGE UPLOAD DETECTION
# ==========================================================
elif menu == "Upload & Detect":

    st.header("📤 Upload Image")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original")

            results = st.session_state.model(image)
            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result")

# ==========================================================
# 4️⃣ REAL TIME CAMERA
# ==========================================================
elif menu == "Real-time Camera":

    st.header("🎥 Real-time Detection")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:

        run = st.checkbox("Start Camera")

        if run:
            cap = cv2.VideoCapture(0)
            frame_window = st.image([])

            while run:
                ret, frame = cap.read()
                if not ret:
                    break

                results = st.session_state.model(frame)
                annotated = results[0].plot()
                frame_window.image(annotated)

            cap.release()

# ==========================================================
# 5️⃣ TRAINING
# ==========================================================
elif menu == "Training":

    st.header("🚀 Train YOLOv8 Model")

    data_yaml = st.text_input("Dataset YAML", "data.yaml")
    epochs = st.slider("Epochs", 10, 300, 50)
    imgsz = st.slider("Image Size", 320, 1280, 640)

    if st.session_state.model is None:
        st.warning("Load base model first.")
    else:
        if st.button("Start Training"):
            st.session_state.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project="runs/detect",
                name="train_custom"
            )
            st.success("Training Complete!")

# ==========================================================
# 6️⃣ AUTO HYPERPARAMETER TUNING
# ==========================================================
elif menu == "Auto Hyperparameter Tuning":

    st.header("⚙ Auto Hyperparameter Tuning")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:

        if st.button("Start Auto Tuning (YOLO Evolution)"):

            st.info("Running Genetic Hyperparameter Evolution...")

            st.session_state.model.tune(
                data="data.yaml",
                epochs=30,
                iterations=50,
                optimizer="AdamW",
                plots=True,
                save=True
            )

            st.success("Hyperparameter tuning completed!")

# ==========================================================
# 7️⃣ EVALUATION DASHBOARD
# ==========================================================
elif menu == "Evaluation Dashboard":

    st.header("📈 Advanced Evaluation")

    run_name = st.text_input("Run Name", "train_custom")
    results_csv = f"runs/detect/{run_name}/results.csv"

    if os.path.exists(results_csv):

        df = pd.read_csv(results_csv)

        st.plotly_chart(px.line(df, y="metrics/mAP50(B)", title="mAP50"))
        st.plotly_chart(px.line(df, y="metrics/recall(B)", title="Recall"))
        st.plotly_chart(px.line(df, y="train/box_loss", title="Loss"))

        final_map = df["metrics/mAP50(B)"].iloc[-1]

        if final_map > 0.85:
            st.success("🔥 Production Ready Model")
        elif final_map > 0.65:
            st.warning("⚡ Needs Fine Tuning")
        else:
            st.error("❌ Improve Dataset / Training")

    else:
        st.warning("Run not found.")

# ==========================================================
# 8️⃣ GRAD-CAM EXPLAINABILITY
# ==========================================================
elif menu == "Grad-CAM Explainability":

    st.header("🧠 Grad-CAM Visualization")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:

        uploaded_file = st.file_uploader("Upload Image for Grad-CAM",
                                         type=["jpg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            img_np = np.array(image)

            results = st.session_state.model(img_np)
            annotated = results[0].plot()

            st.image(annotated, caption="Detection")

            st.info("""
            Grad-CAM highlights image regions that influenced detection.
            (For full GradCAM implementation integrate pytorch-grad-cam)
            """)
