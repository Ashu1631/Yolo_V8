import streamlit as st
import os
import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import cv2
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(layout="wide", page_title="YOLOv8 ML Dashboard")

st.title("🚀 YOLOv8 Production ML Dashboard")

# -------------------- TABS --------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 Dataset Viewer",
    "🏋️ Training",
    "📊 Evaluation",
    "⚖ Model Compare",
    "🔍 Prediction",
    "🖥 GPU Monitor"
])

# -------------------- DATASET VIEWER --------------------

with tab1:
    st.header("Dataset Annotation Viewer")

    image_folder = "data/train/images"
    label_folder = "data/train/labels"

    if os.path.exists(image_folder):
        images = os.listdir(image_folder)
        selected = st.selectbox("Select Image", images)

        if selected:
            img_path = os.path.join(image_folder, selected)
            label_path = os.path.join(label_folder, selected.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            h, w, _ = img.shape

            if os.path.exists(label_path):
                with open(label_path) as f:
                    lines = f.readlines()

                for line in lines:
                    cls, x, y, bw, bh = map(float, line.split())
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)

            st.image(img, channels="BGR")
    else:
        st.warning("Dataset not found")

# -------------------- TRAINING --------------------

with tab2:
    st.header("Train YOLO Model")

    epochs = st.slider("Epochs", 10, 200, 50)

    if st.button("Start Training"):
        model = YOLO("yolov8n.pt")
        model.train(data="data.yaml", epochs=epochs, imgsz=640)
        st.success("Training Completed!")

# -------------------- EVALUATION --------------------

with tab3:
    st.header("Model Evaluation")

    csv_path = "runs/detect/train/results.csv"
    cm_csv = "runs/detect/train/confusion_matrix.csv"
    cm_img = "runs/detect/train/confusion_matrix.png"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loss Curve")
            st.line_chart(df[["train/box_loss", "val/box_loss"]])

        with col2:
            st.subheader("Precision / Recall")
            st.line_chart(df[["metrics/precision(B)", "metrics/recall(B)"]])

        # Interactive Confusion Matrix
        if os.path.exists(cm_csv):
            st.subheader("Interactive Confusion Matrix")
            cm = pd.read_csv(cm_csv, header=None)
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(cm_img):
        st.image(cm_img)

    # PDF REPORT
    if st.button("Download PDF Report"):
        pdf_path = "evaluation_report.pdf"
        doc = SimpleDocTemplate(pdf_path)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("YOLOv8 Evaluation Report", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        if os.path.exists(csv_path):
            table_data = df.tail(1).values.tolist()
            headers = list(df.columns)
            table = Table([headers] + table_data)
            elements.append(table)

        doc.build(elements)
        st.success("PDF Generated")

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="evaluation_report.pdf")

# -------------------- MODEL COMPARISON --------------------

with tab4:
    st.header("Model Version Comparison")

    if os.path.exists("models"):
        model_files = os.listdir("models")

        if len(model_files) >= 2:
            m1 = st.selectbox("Model 1", model_files)
            m2 = st.selectbox("Model 2", model_files)

            if st.button("Compare Models"):
                model1 = YOLO(f"models/{m1}")
                model2 = YOLO(f"models/{m2}")

                r1 = model1.val(data="data.yaml")
                r2 = model2.val(data="data.yaml")

                st.write("Model 1 mAP:", r1.box.map)
                st.write("Model 2 mAP:", r2.box.map)
        else:
            st.info("Add at least two models inside models/ folder")

# -------------------- PREDICTION --------------------

with tab5:
    st.header("Image Prediction")

    model_path = "runs/detect/train/weights/best.pt"

    if os.path.exists(model_path):
        model = YOLO(model_path)

        uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if uploaded:
            results = model(uploaded)
            result_img = results[0].plot()
            st.image(result_img)
    else:
        st.warning("Train model first")

# -------------------- GPU MONITOR --------------------

with tab6:
    st.header("GPU Monitoring")

    if torch.cuda.is_available():
        st.success("GPU Available")
        st.write("GPU Name:", torch.cuda.get_device_name(0))
        st.write("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,2),"GB")
        st.write("Memory Reserved:", round(torch.cuda.memory_reserved(0)/1024**3,2),"GB")
    else:
        st.error("GPU Not Available")
