import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard", layout="wide")

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio(
    "Go To",
    [
        "Model Selection",
        "Dataset Viewer",
        "Upload & Detect",
        "Training",
        "Evaluation Dashboard",
        "Model Comparison"
    ]
)

# ==========================
# SESSION STATE
# ==========================
if "model" not in st.session_state:
    st.session_state.model = None

if "run_name" not in st.session_state:
    st.session_state.run_name = "train"

# ==========================================================
# 1️⃣ MODEL SELECTION
# ==========================================================
if menu == "Model Selection":

    st.header("📊 Model Selection")

    model_options = ["yolov8n.pt", "yolov8s.pt", "best.pt"]
    selected_model = st.selectbox("Select Model", model_options)

    if st.button("Load Model"):
        st.session_state.model = YOLO(selected_model)
        st.success(f"{selected_model} loaded successfully..")

# ==========================================================
# 2️⃣ DATASET VIEWER
# ==========================================================
elif menu == "Dataset Viewer":

    st.header("🖼 Dataset Viewer")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:
        image_folder = "data/images"

        if os.path.exists(image_folder):
            images = os.listdir(image_folder)
            selected_image = st.selectbox("Select Image", images)
            image_path = os.path.join(image_folder, selected_image)

            st.image(image_path, caption="Original")

            if st.button("Run Detection"):
                results = st.session_state.model(image_path)
                annotated = results[0].plot()
                st.image(annotated, caption="Detected")

# ==========================================================
# 3️⃣ UPLOAD IMAGE / VIDEO
# ==========================================================
elif menu == "Upload & Detect":

    st.header("📤 Upload Image / Video")

    if st.session_state.model is None:
        st.warning("Load model first.")
    else:
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        )

        if uploaded_file is not None:

            if "image" in uploaded_file.type:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                results = st.session_state.model(image)
                annotated = results[0].plot()
                st.image(annotated)

            elif "video" in uploaded_file.type:
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.video(temp_path)

                if st.button("Run Video Detection"):
                    st.session_state.model(temp_path, save=True)
                    st.success("Video processed.")

# ==========================================================
# 4️⃣ TRAINING SECTION (AUTO TRAIN TRIGGER)
# ==========================================================
elif menu == "Training":

    st.header("🚀 Train Model from Dashboard")

    data_yaml = st.text_input("Dataset YAML Path", "data.yaml")
    epochs = st.number_input("Epochs", min_value=1, max_value=300, value=50)
    imgsz = st.number_input("Image Size", min_value=320, max_value=1280, value=640)
    run_name = st.text_input("Run Name", "train_custom")

    if st.session_state.model is None:
        st.warning("Load base model first.")
    else:
        if st.button("Start Training"):

            st.info("Training started... This may take time.")

            st.session_state.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                project="runs/detect",
                name=run_name
            )

            st.session_state..run_name = run_name
            st.success("Training completed.")

# ==========================================================
# 5️⃣ EVALUATION DASHBOARD
# ==========================================================
elif menu == "Evaluation Dashboard":

    st.header("📈 Evaluation Dashboard")

    run_name = st.text_input("Run Folder Name", st.session_state.run_name)
    results_csv = f"runs/detect/{run_name}/results.csv"

    if os.path.exists(results_csv):

        df = pd.read_csv(results_csv)

        # Loss
        st.plotly_chart(px.line(df, y="train/box_loss", title="Loss Curve"))

        # Recall
        st.plotly_chart(px.line(df, y="metrics/recall(B)", title="Recall Curve"))

        # mAP50
        st.plotly_chart(px.line(df, y="metrics/mAP50(B)", title="mAP50"))

        # mAP50-95
        st.plotly_chart(px.line(df, y="metrics/mAP50-95(B)", title="mAP50-95"))

        # F1
        df["f1"] = 2 * (
            df["metrics/precision(B)"] * df["metrics/recall(B)"]
        ) / (
            df["metrics/precision(B)"] + df["metrics/recall(B)"]
        )

        st.plotly_chart(px.line(df, y="f1", title="F1 Score"))

        # ==========================
        # REAL CONFUSION MATRIX
        # ==========================
        st.subheader("🧠 Real Confusion Matrix")

        model = YOLO(f"runs/detect/{run_name}/weights/best.pt")
        metrics = model.val(data="data.yaml", save_json=True)

        cm = metrics.confusion_matrix.matrix
        class_names = metrics.names

        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        fig_cm = px.imshow(
            cm_norm,
            text_auto=True,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            title="Normalized Confusion Matrix"
        )

        st.plotly_chart(fig_cm)

        # ==========================
        # AUTO INTERPRETATION
        # ==========================
        final_map = df["metrics/mAP50(B)"].iloc[-1]
        final_recall = df["metrics/recall(B)"].iloc[-1]
        final_f1 = df["f1"].iloc[-1]

        if final_map > 0.80:
            status = "Excellent"
        elif final_map > 0.60:
            status = "Moderate"
        else:
            status = "Needs Improvement"

        st.success(f"""
        mAP50: {final_map:.2f}  
        Recall: {final_recall:.2f}  
        F1: {final_f1:.2f}  

        Overall Performance: {status}
        """)

        # ==========================
        # PDF REPORT
        # ==========================
        if st.button("Download PDF Report"):

            pdf_path = "evaluation_report.pdf"
            doc = SimpleDocTemplate(pdf_path)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("YOLOv8 Evaluation Report", styles["Heading1"]))
            elements.append(Spacer(1, 0.3 * inch))

            table_data = [
                ["Metric", "Value"],
                ["mAP50", f"{final_map:.2f}"],
                ["Recall", f"{final_recall:.2f}"],
                ["F1 Score", f"{final_f1:.2f}"],
                ["Performance", status]
            ]

            elements.append(Table(table_data))
            doc.build(elements)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Click to Download",
                    f,
                    file_name="evaluation_report.pdf"
                )

    else:
        st.warning("Run results not found.")

# ==========================================================
# 6️⃣ MODEL COMPARISON
# ==========================================================
elif menu == "Model Comparison":

    st.header("📊 Model Version Comparison")

    run1 = st.text_input("Run 1 Name", "train")
    run2 = st.text_input("Run 2 Name", "train_custom")

    file1 = f"runs/detect/{run1}/results.csv"
    file2 = f"runs/detect/{run2}/results.csv"

    if os.path.exists(file1) and os.path.exists(file2):

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df1["metrics/mAP50(B)"],
            mode="lines",
            name=run1
        ))
        fig.add_trace(go.Scatter(
            y=df2["metrics/mAP50(B)"],
            mode="lines",
            name=run2
        ))

        fig.update_layout(title="mAP50 Comparison")
        st.plotly_chart(fig)

    else:
        st.warning("Run files not found.")
