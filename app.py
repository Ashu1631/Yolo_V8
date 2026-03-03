import streamlit as st
import os
import cv2
import time
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ==========================================================
# LOGIN
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", {})
else:
    users = {}

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")
    st.stop()

# ==========================================================
# SESSION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# NAVIGATION (GREEN SELECTED, RED OTHERS)
# ==========================================================
st.sidebar.markdown("## 🚀 Navigation")

pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases",
    "Model Comparison"
]

selected_page = st.sidebar.radio(
    "",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected_page

st.markdown("""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: #dc3545;
    padding: 8px;
    margin-bottom: 6px;
    border-radius: 8px;
    color: white;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #28a745 !important;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

# ==========================================================
# FAILURE EXTRACTION LOGIC
# ==========================================================
def extract_failures(results, image, filename):
    os.makedirs("failure_cases/false_positive", exist_ok=True)
    os.makedirs("failure_cases/false_negative", exist_ok=True)

    boxes = results[0].boxes

    # False Negative (no detection)
    if boxes is None or len(boxes) == 0:
        cv2.imwrite(f"failure_cases/false_negative/{filename}", image)
        return

    # False Positive (low confidence)
    conf = boxes.conf.cpu().numpy()
    if any(conf < 0.30):
        cv2.imwrite(f"failure_cases/false_positive/{filename}", image)

# ==========================================================
# MODEL SELECTION
# ==========================================================
if page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model", ["-- Select --"] + models)
    if selected != "-- Select --":
        st.session_state.model = YOLO(selected)
        st.success("Model Loaded Successfully ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==========================================================
# UPLOAD
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model
    uploaded = st.file_uploader("Upload Image/Video",
                                type=["jpg", "png", "jpeg"])

    if uploaded:
        img = cv2.imread(uploaded.name)
        results = model(img)
        annotated = results[0].plot()
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        # Auto failure extraction
        extract_failures(results, img, uploaded.name)

# ==========================================================
# EVALUATION (FULL YOLO GRAPHS)
# ==========================================================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    if os.path.exists("analysis/results.csv"):

        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("mAP50",
                    f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("mAP50-95",
                    f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("Precision",
                    f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("Recall",
                    f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("Training Loss")
        st.line_chart(df[[
            'train/box_loss',
            'train/cls_loss',
            'train/dfl_loss'
        ]])

        st.subheader("Validation Loss")
        st.line_chart(df[[
            'val/box_loss',
            'val/cls_loss',
            'val/dfl_loss'
        ]])

        st.subheader("Precision / Recall / mAP")
        st.line_chart(df[[
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)'
        ]])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.image("analysis/confusion_matrix.png")

# ==========================================================
# FAILURE CASES VIEWER
# ==========================================================
if page == "Failure Cases":

    st.title("⚠ Failure Cases")

    categories = ["false_positive", "false_negative"]

    selected_cat = st.selectbox("Select Failure Type", categories)

    folder = f"failure_cases/{selected_cat}"

    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            selected_img = st.selectbox("Select Image", files)
            st.image(os.path.join(folder, selected_img))
        else:
            st.info("No images found.")
    else:
        st.info("No failure data available.")
