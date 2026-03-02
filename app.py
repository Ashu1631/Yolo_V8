import streamlit as st
import pandas as pd
import os
import plotly.express as px
from ultralytics import YOLO
from PIL import Image

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard", layout="wide")

st.sidebar.title("🚀 YOLOv8 Enterprise Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Model Selection",
        "Upload & Detect",
        "Evaluation Dashboard"
    ]
)

# ==========================================================
# SESSION STATE
# ==========================================================
if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# 1️⃣ MODEL SELECTION
# ==========================================================
if menu == "Model Selection":

    st.header("📦 Model Selection")

    # Default official YOLO models
    default_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    # Absolute path for root best.pt
    root_best_path = os.path.join(os.getcwd(), "best.pt")

    model_options = default_models.copy()

    # Add root best.pt if exists
    if os.path.isfile(root_best_path):
        model_options.append("best.pt")

    model_choice = st.selectbox("Select Model", model_options)

    if st.button("Load Model"):

        try:
            if model_choice == "best.pt":
                model_path = root_best_path
            else:
                model_path = model_choice

            st.session_state.model = YOLO(model_path)
            st.success(f"Model loaded successfully: {model_choice}")

        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")

    # Debug info (can remove later)
    st.write("Current Directory:", os.getcwd())
    st.write("Available Files:", os.listdir())

# ==========================================================
# 2️⃣ UPLOAD & DETECT
# ==========================================================
elif menu == "Upload & Detect":

    st.header("📤 Upload Image for Detection")

    if st.session_state.model is None:
        st.warning("Please load a model first.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image")

            results = st.session_state.model(image)
            annotated = results[0].plot()

            st.image(annotated, caption="Detection Result")

# ==========================================================
# 3️⃣ EVALUATION DASHBOARD
# ==========================================================
elif menu == "Evaluation Dashboard":

    st.header("📈 Evaluation Dashboard")

    runs_path = "runs/detect"

    if not os.path.exists(runs_path):
        st.info("No runs/detect folder found.")
    else:

        run_folders = os.listdir(runs_path)

        if not run_folders:
            st.info("No training runs available.")
        else:

            selected_run = st.selectbox("Select Run Folder", run_folders)
            run_path = os.path.join(runs_path, selected_run)

            results_csv = os.path.join(run_path, "results.csv")
            results_png = os.path.join(run_path, "results.png")
            pr_curve = os.path.join(run_path, "BoxPR_curve.png")
            f1_curve = os.path.join(run_path, "BoxF1_curve.png")
            confusion_png = os.path.join(run_path, "confusion_matrix.png")
            best_model = os.path.join(run_path, "weights", "best.pt")

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

                st.plotly_chart(
                    px.line(df,
                            y="metrics/mAP50(B)",
                            title="mAP50 Curve")
                )

                st.plotly_chart(
                    px.line(df,
                            y="train/box_loss",
                            title="Loss Curve")
                )

            else:
                st.warning("results.csv not found.")

            # Show images
            st.subheader("📊 Training Visualizations")

            if os.path.exists(results_png):
                st.image(results_png, caption="Results Overview")

            if os.path.exists(pr_curve):
                st.image(pr_curve, caption="Precision-Recall Curve")

            if os.path.exists(f1_curve):
                st.image(f1_curve, caption="F1 Curve")

            if os.path.exists(confusion_png):
                st.image(confusion_png, caption="Confusion Matrix")

            # Download trained best model
            if os.path.exists(best_model):
                with open(best_model, "rb") as f:
                    st.download_button(
                        "⬇ Download Trained best.pt",
                        f,
                        file_name="best.pt"
                    )
