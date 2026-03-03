import streamlit as st
import os
import cv2
import time
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ==========================================================
# LOAD USERS FROM YAML
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users_yaml = yaml.safe_load(f)
        users = users_yaml.get("users", {})
else:
    users = {}

# ==========================================================
# LOGIN SYSTEM
# ==========================================================
if not st.session_state.logged_in:

    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
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
# NAVIGATION
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

for p in pages:
    if st.session_state.page == p:
        st.sidebar.markdown(
            f"<div style='background:#28a745;padding:5px;border-radius:6px;color:white;margin-bottom:3px;'>👉 {p}</div>",
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=p):
            st.session_state.page = p
            st.rerun()

st.sidebar.markdown("""
<style>
div[data-testid="stButton"] button {
    background:#dc3545;
    color:white;
    padding:5px;
    border-radius:6px;
    margin-bottom:3px;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

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
# UPLOAD & DATASET
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    # Upload
    with tab1:

        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg","png","jpeg","mp4"])

        if uploaded:

            compare = st.checkbox("Enable Comparison (best.pt vs yolov8n.pt)")

            temp_path = uploaded.name
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            if temp_path.endswith(("jpg","png","jpeg")):

                img = cv2.imread(temp_path)
                start = time.time()

                if compare:
                    col1, col2 = st.columns(2)

                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)

                    fps = 1 / (time.time() - start)

                    col1.markdown(f"### 🟢 best.pt | FPS: {fps:.2f}")
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown(f"### 🔵 yolov8n.pt | FPS: {fps:.2f}")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    fps = 1 / (time.time() - start)
                    annotated = r[0].plot()
                    cv2.putText(annotated, f"FPS: {fps:.2f}",
                                (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(0,255,0),2)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # Dataset
    with tab2:

        dataset_path = "datasets"

        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.endswith(("jpg","png","jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":
                img = cv2.imread(os.path.join(dataset_path, selected_img))
                r = model(img)
                st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))

# ==========================================================
# WEBCAM LIVE DETECTION
# ==========================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            start = time.time()
            results = self.model(img)
            annotated = results[0].plot()
            fps = 1 / (time.time() - start)

            cv2.putText(annotated,
                        f"FPS: {fps:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )

    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==========================================================
# EVALUATION
# ==========================================================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    csv_path = "analysis/results.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col3.metric("Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("Loss Curve")
        st.area_chart(df[['train/box_loss',
                          'train/cls_loss',
                          'train/dfl_loss']])
