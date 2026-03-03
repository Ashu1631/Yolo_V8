import streamlit as st
import os
import cv2
import time
import yaml
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ==========================================================
# LOGIN SYSTEM
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f)["users"]
else:
    users = {"admin": {"password": "admin"}}

def hash_pass(p):
    return hashlib.sha256(p.encode()).hexdigest()

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users:
            stored = users[username]["password"]
            if stored == password or stored == hash_pass(password):
                st.session_state.logged_in = True
                st.rerun()
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
# NAVIGATION (Compact + Green/Red)
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
            f"""
            <div style="background:#28a745;
                        padding:4px 6px;
                        border-radius:5px;
                        color:white;
                        font-size:14px;
                        font-weight:600;
                        margin-bottom:2px;">
                👉 {p}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

st.sidebar.markdown("""
<style>
div[data-testid="stButton"] button {
    background:#dc3545;
    color:white;
    padding:4px 6px;
    border-radius:5px;
    margin-bottom:2px;
    font-size:14px;
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
# FAILURE EXTRACTION
# ==========================================================
def extract_failures(results, image, threshold=0.3):
    os.makedirs("failure_cases", exist_ok=True)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.imwrite("failure_cases/no_detection.jpg", image)
        return
    conf = boxes.conf.cpu().numpy()
    if any(conf < threshold):
        cv2.imwrite("failure_cases/low_confidence.jpg", image)

# ==========================================================
# UPLOAD & DATASET
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

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

                if compare:
                    col1, col2 = st.columns(2)

                    col1.markdown("### 🟢 best.pt")
                    r1 = YOLO("best.pt")(img)
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown("### 🔵 yolov8n.pt")
                    r2 = YOLO("yolov8n.pt")(img)
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    extract_failures(r, img)
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))

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
# WEBCAM
# ==========================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    )

    model = st.session_state.model

    class Processor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=Processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )

# ==========================================================
# EVALUATION DASHBOARD
# ==========================================================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    csv_path = "analysis/results.csv"

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("📊 mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("📊 mAP50-95", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("🎯 Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("🔁 Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("📉 Loss Curve")
        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])

        st.subheader("📈 Recall Curve")
        st.line_chart(df[['metrics/recall(B)']])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.subheader("🔥 Confusion Matrix")
            st.image("analysis/confusion_matrix.png")

# ==========================================================
# FAILURE CASES
# ==========================================================
if page == "Failure Cases":
    st.title("⚠ Failure Cases")
    if os.path.exists("failure_cases"):
        files = os.listdir("failure_cases")
        if files:
            selected = st.selectbox("Select Failure Case", files)
            st.image(os.path.join("failure_cases", selected))
        else:
            st.info("No failure cases found.")

# ==========================================================
# MODEL COMPARISON
# ==========================================================
if page == "Model Comparison":

    st.title("🚀 Model Comparison")

    if os.path.exists("analysis/results.csv"):

        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        metrics = {
            "mAP50": latest['metrics/mAP50(B)'],
            "Precision": latest['metrics/precision(B)'],
            "Recall": latest['metrics/recall(B)']
        }

        labels = list(metrics.keys())
        values = list(metrics.values())

        st.subheader("📈 Line Chart")
        st.line_chart(pd.DataFrame(metrics, index=[0]))

        st.subheader("📊 Bar Chart")
        st.plotly_chart(px.bar(x=labels, y=values))

        st.subheader("🥧 Pie Chart")
        st.plotly_chart(px.pie(names=labels, values=values))

        st.subheader("📊 Histogram")
        st.plotly_chart(px.histogram(df, x='metrics/mAP50(B)'))

        st.subheader("💧 Waterfall Chart")
        fig = go.Figure(go.Waterfall(
            x=labels,
            y=values,
            measure=["relative","relative","relative"]
        ))
        st.plotly_chart(fig)

        st.subheader("🔻 Funnel Chart")
        st.plotly_chart(px.funnel(x=values, y=labels))
