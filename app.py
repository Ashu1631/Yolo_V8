import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="YOLOv8 Enterprise AI Platform", layout="wide")

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
        "Evaluation Dashboard"
    ]
)

# ==========================================================
# SESSION STATE
# ==========================================================
if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# 1️⃣ MODEL SELECTION + DATASET VIEWER
# ==========================================================
if menu == "Model Selection":

    st.header("📦 Model Selection & Dataset Detection")

    model_choice = st.selectbox(
        "Select Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "best.pt"]
    )

    if st.button("Load Model"):
        st.session_state.model = YOLO(model_choice)
        st.success("Model Loaded Successfully!")

    # Dataset image dropdown
    if st.session_state.model is not None:

        image_folder = "data/images"

        if os.path.exists(image_folder):

            images = [
                img for img in os.listdir(image_folder)
                if img.lower().endswith((".jpg", ".png", ".jpeg"))
            ]

            if images:
                selected_image = st.selectbox("Select Dataset Image", images)

                image_path = os.path.join(image_folder, selected_image)
                st.image(image_path, caption="Original Image")

                if st.button("Run Detection"):
                    results = st.session_state.model(image_path)
                    annotated = results[0].plot()
                    st.image(annotated, caption="Detection Result")

        else:
            st.warning("data/images folder not found.")

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

        if class_counts:
            df = pd.DataFrame({
                "Class": list(class_counts.keys()),
                "Count": list(class_counts.values())
            })

            st.plotly_chart(px.pie(df, names="Class", values="Count",
                                   title="Class Distribution"))

            st.plotly_chart(px.bar(df, x="Class", y="Count",
                                   title="Class Imbalance"))

        else:
            st.warning("No labels found.")

    else:
        st.warning("data/labels folder not found.")

# ==========================================================
# 3️⃣ UPLOAD IMAGE DETECTION
# ==========================================================
elif menu == "Upload & Detect":

    st.header("📤 Upload Image Detection")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:
        uploaded_file = st.file_uploader("Upload Image",
                                         type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image")

            results = st.session_state.model(image)
            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result")

# ==========================================================
# 4️⃣ REAL TIME CAMERA
# ==========================================================
elif menu == "Real-time Camera":

    st.header("🎥 Real-time Object Detection")

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

    data_yaml = st.text_input("Dataset YAML Path", "data.yaml")
    epochs = st.slider("Epochs", 10, 300, 50)
    imgsz = st.slider("Image Size", 320, 1280, 640)
    run_name = st.text_input("Run Name", "train_custom")

    if st.session_state.model is None:
        st.warning("Load base model first.")
    else:
        if st.button("Start Training"):
            st.session_state.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project="runs/detect",
                name=run_name
            )
            st.success("Training Completed!")

# ==========================================================
# 6️⃣ AUTO HYPERPARAMETER TUNING
# ==========================================================
elif menu == "Auto Hyperparameter Tuning":

    st.header("⚙ Auto Hyperparameter Evolution")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:
        if st.button("Start Auto Tuning"):
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

    st.header("📈 Complete Experiment Dashboard")

    run_name = st.text_input("Run Folder Name", "train_custom")
    run_path = f"runs/detect/{run_name}"

    results_csv = os.path.join(run_path, "results.csv")
    results_png = os.path.join(run_path, "results.png")
    pr_curve = os.path.join(run_path, "BoxPR_curve.png")
    f1_curve = os.path.join(run_path, "BoxF1_curve.png")
    confusion_png = os.path.join(run_path, "confusion_matrix.png")
    best_model = os.path.join(run_path, "weights/best.pt")

    # CSV Metrics
    if os.path.exists(results_csv):

        df = pd.read_csv(results_csv)

        col1, col2, col3 = st.columns(3)

        col1.metric("Final mAP50",
                    f"{df['metrics/mAP50(B)'].iloc[-1]:.2f}")
        col2.metric("Final Recall",
                    f"{df['metrics/recall(B)'].iloc[-1]:.2f}")
        col3.metric("Final Precision",
                    f"{df['metrics/precision(B)'].iloc[-1]:.2f}")

        st.plotly_chart(px.line(df, y="metrics/mAP50(B)",
                                title="mAP50 Curve"))

        st.plotly_chart(px.line(df, y="train/box_loss",
                                title="Loss Curve"))

    else:
        st.warning("results.csv not found.")

    # Show YOLO generated visual files
    st.subheader("📊 Training Visualizations")

    if os.path.exists(results_png):
        st.image(results_png, caption="Results Overview")

    if os.path.exists(pr_curve):
        st.image(pr_curve, caption="Precision-Recall Curve")

    if os.path.exists(f1_curve):
        st.image(f1_curve, caption="F1 Curve")

    if os.path.exists(confusion_png):
        st.image(confusion_png, caption="Confusion Matrix")

    # Download best model
    if os.path.exists(best_model):
        with open(best_model, "rb") as f:
            st.download_button(
                "⬇ Download best.pt",
                f,
                file_name="best.pt"
            )
