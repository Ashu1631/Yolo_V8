import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import time
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="YOLOv8 Enterprise", layout="wide")

# ==================================================
# SESSION
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

if "model" not in st.session_state:
    st.session_state.model = None

# ==================================================
# NAVIGATION (compact + spacing fixed)
# ==================================================
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
            f"""
            <div style="
                background:#28a745;
                padding:6px;
                border-radius:6px;
                color:white;
                font-weight:600;
                margin-bottom:3px;">
                👉 {p}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

        st.sidebar.markdown(
            """
            <style>
            div[data-testid="stButton"] button {
                background:#dc3545;
                color:white;
                margin-bottom:3px;
                border-radius:6px;
                padding:5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

page = st.session_state.page

# ==================================================
# MODEL SELECTION
# ==================================================
if page == "Model Selection":

    st.title("📦 Model Selection")

    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":
        st.session_state.model = YOLO(selected)
        st.success("Model Loaded ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==================================================
# WEBCAM FIX (best possible cloud solution)
# ==================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    model = st.session_state.model

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            start = time.time()
            results = model(img)
            annotated = results[0].plot()
            fps = 1 / (time.time() - start)

            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ==================================================
# EVALUATION DASHBOARD
# ==================================================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    csv_path = "analysis/results.csv"

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]

        st.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        st.metric("Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        st.metric("Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("Loss Curve")
        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])

        st.subheader("Recall Curve")
        st.line_chart(df[['metrics/recall(B)']])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.image("analysis/confusion_matrix.png")

    else:
        st.warning("results.csv not found.")

# ==================================================
# FAILURE CASES
# ==================================================
if page == "Failure Cases":

    st.title("⚠ Failure Cases")

    failure_dir = "failure_cases"

    if os.path.exists(failure_dir):

        files = os.listdir(failure_dir)

        if files:
            selected = st.selectbox("Select Failure Case", files)
            st.image(os.path.join(failure_dir, selected))
        else:
            st.info("No failure images found.")
    else:
        st.info("Failure folder not found.")

# ==================================================
# MODEL COMPARISON (ALL CHARTS ADDED)
# ==================================================
if page == "Model Comparison":

    st.title("📊 Model Comparison Advanced")

    csv_path = "analysis/results.csv"

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        st.subheader("Line Chart")
        st.line_chart(df[['metrics/mAP50(B)','metrics/precision(B)','metrics/recall(B)']])

        st.subheader("Area Chart")
        st.area_chart(df[['metrics/mAP50(B)','metrics/precision(B)','metrics/recall(B)']])

        st.subheader("Histogram")
        st.plotly_chart(px.histogram(df, x='metrics/mAP50(B)'))

        st.subheader("Heatmap")
        corr = df.corr()
        st.plotly_chart(px.imshow(corr, text_auto=True))

        latest = df.iloc[-1]

        metrics = [
            latest['metrics/mAP50(B)'],
            latest['metrics/precision(B)'],
            latest['metrics/recall(B)']
        ]

        labels = ["mAP50","Precision","Recall"]

        st.subheader("Bar Chart")
        st.plotly_chart(px.bar(x=labels, y=metrics))

        st.subheader("Pie Chart")
        st.plotly_chart(px.pie(names=labels, values=metrics))

        st.subheader("Waterfall Chart")

        fig = go.Figure(go.Waterfall(
            x=labels,
            y=metrics,
            measure=["relative","relative","relative"],
        ))

        st.plotly_chart(fig)

    else:
        st.warning("results.csv not found.")
