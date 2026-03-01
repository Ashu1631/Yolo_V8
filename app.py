import streamlit as st
import os
import yaml
import pandas as pd
import torch
from ultralytics import YOLO
import plotly.express as px
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(layout="wide", page_title="YOLOv8 Enterprise Dashboard")

# ================= LOGIN SYSTEM =================

def load_users():
    with open("users.yaml", "r") as file:
        return yaml.safe_load(file)["users"]

def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    users = load_users()

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
        else:
            st.sidebar.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ================= HEADER =================

st.title("🚀 YOLOv8 Enterprise ML Dashboard")
st.sidebar.success(f"Logged in as {st.session_state['user']}")

# ================= PATHS =================

history_path = "experiments/history.csv"
os.makedirs("experiments", exist_ok=True)

# ================= TABS =================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏋️ Train",
    "📊 Evaluation",
    "📈 Version History",
    "🔍 Prediction",
    "🖥 GPU Monitor"
])

# ================= TRAINING =================

with tab1:
    st.header("Train Model")

    epochs = st.slider("Epochs", 10, 200, 50)

    if st.button("Start Training"):
        model = YOLO("yolov8n.pt")
        model.train(data="data.yaml", epochs=epochs, imgsz=640)

        st.success("Training Completed")

        # Read metrics
        results_csv = "runs/detect/train/results.csv"

        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]

            new_entry = {
                "Version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Epochs": epochs,
                "mAP50": last_row["metrics/mAP50(B)"],
                "Precision": last_row["metrics/precision(B)"],
                "Recall": last_row["metrics/recall(B)"],
                "Train_Box_Loss": last_row["train/box_loss"],
                "Val_Box_Loss": last_row["val/box_loss"]
            }

            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                history_df = pd.concat([history_df, pd.DataFrame([new_entry])])
            else:
                history_df = pd.DataFrame([new_entry])

            history_df.to_csv(history_path, index=False)
            st.success("Version Logged Successfully")

# ================= EVALUATION =================

with tab2:
    st.header("Current Model Evaluation")

    results_csv = "runs/detect/train/results.csv"

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Loss Curve")
            st.line_chart(df[["train/box_loss", "val/box_loss"]])

        with col2:
            st.subheader("Precision / Recall")
            st.line_chart(df[["metrics/precision(B)", "metrics/recall(B)"]])

# ================= VERSION HISTORY =================

with tab3:
    st.header("Model Version Tracking")

    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)

        st.subheader("Version Table")
        st.dataframe(history_df, use_container_width=True)

        st.subheader("mAP50 Trend")
        fig = px.line(
            history_df,
            x="Version",
            y="mAP50",
            markers=True,
            title="mAP50 History Over Versions"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Loss Trend")
        fig2 = px.line(
            history_df,
            x="Version",
            y=["Train_Box_Loss", "Val_Box_Loss"],
            markers=True,
            title="Loss Trend"
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No training history yet.")

# ================= PREDICTION =================

with tab4:
    st.header("Prediction")

    model_path = "runs/detect/train/weights/best.pt"

    if os.path.exists(model_path):
        model = YOLO(model_path)
        uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if uploaded:
            results = model(uploaded)
            result_img = results[0].plot()
            st.image(result_img)
    else:
        st.warning("Train model first.")

# ================= GPU MONITOR =================

with tab5:
    st.header("GPU Monitoring")

    if torch.cuda.is_available():
        st.success("GPU Available")
        st.write("GPU Name:", torch.cuda.get_device_name(0))
        st.write("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,2),"GB")
        st.write("Memory Reserved:", round(torch.cuda.memory_reserved(0)/1024**3,2),"GB")
    else:
        st.error("GPU Not Available")
