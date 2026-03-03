import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import hashlib
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ==========================================================
# SESSION STATE
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# LOGIN SYSTEM
# ==========================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k: v["password"] for k, v in users_yaml["users"].items()}

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users:
            if users[username] == password or users[username] == hash_password(password):
                st.session_state.logged_in = True
                st.rerun()
        st.error("Invalid Credentials ❌")
    st.stop()

# ==========================================================
# SIDEBAR NAVIGATION (BLUE ACTIVE)
# ==========================================================
st.sidebar.markdown("## 🚀 Navigation")

pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases"
]

for p in pages:
    if st.session_state.page == p:
        st.sidebar.markdown(
            f"""
            <div style="
                background:#1f77ff;
                padding:10px;
                border-radius:8px;
                color:white;
                font-weight:bold;
            ">
                👉 {p}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=p):
            st.session_state.page = p
            st.rerun()

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
        st.success(f"{selected} Loaded Successfully ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==========================================================
# HELPER FUNCTION
# ==========================================================
def get_counts(results):
    boxes = results[0].boxes
    counts = {}
    if boxes is not None and len(boxes.cls) > 0:
        names = results[0].names
        for c in boxes.cls.cpu().numpy():
            label = names[int(c)]
            counts[label] = counts.get(label, 0) + 1
    return counts

# ==========================================================
# UPLOAD & DATASET
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model

    tab1, tab2 = st.tabs(["📤 Upload File 🔄", "📂 Dataset Folder 📊"])

    # ---------------- UPLOAD TAB ----------------
    with tab1:
        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg", "png", "jpeg", "mp4"])

        if uploaded:

            path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.read())

            # IMAGE
            if uploaded.name.lower().endswith(("jpg", "png", "jpeg")):
                with st.spinner("🔍 Detecting..."):
                    results = model(path)

                img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(img)

                st.json(get_counts(results))

                if st.checkbox("Compare Models (Image)"):
                    m1 = YOLO("best.pt")
                    m2 = YOLO("yolov8n.pt")

                    r1 = m1(path)
                    r2 = m2(path)

                    col1, col2 = st.columns(2)
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB),
                               caption="best.pt")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB),
                               caption="yolov8n.pt")

            # VIDEO
            if uploaded.name.lower().endswith("mp4"):
                compare = st.checkbox("Compare Models (Video)")

                cap = cv2.VideoCapture(path)
                frame_window = st.empty()

                m1 = YOLO("best.pt")
                m2 = YOLO("yolov8n.pt")

                with st.spinner("🎥 Processing Video..."):
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if compare:
                            r1 = m1(frame)
                            r2 = m2(frame)
                            f1 = r1[0].plot()
                            f2 = r2[0].plot()
                            combined = np.hstack((f1, f2))
                            frame_window.image(
                                cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                            )
                        else:
                            r = model(frame)
                            annotated = r[0].plot()
                            frame_window.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            )

                    cap.release()

    # ---------------- DATASET TAB ----------------
    with tab2:
        dataset_path = "datasets"

        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":
                img_path = os.path.join(dataset_path, selected_img)

                with st.spinner("📊 Running Dataset Detection..."):
                    results = model(img_path)

                img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(img)

                if st.checkbox("Compare Models (Dataset)"):
                    m1 = YOLO("best.pt")
                    m2 = YOLO("yolov8n.pt")

                    r1 = m1(img_path)
                    r2 = m2(img_path)

                    col1, col2 = st.columns(2)
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB),
                               caption="best.pt")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB),
                               caption="yolov8n.pt")

# ==========================================================
# WEBCAM (Most Stable Possible)
# ==========================================================
class WebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO("best.pt")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

if page == "Webcam Detection":
    st.title("📷 Webcam Detection")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": 15},
            "audio": False
        },
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

        col1, col2, col3 = st.columns(3)
        col1.metric("📊 mAP50",
                    f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("🎯 Precision",
                    f"{latest['metrics/precision(B)']*100:.2f}%")
        col3.metric("🔁 Recall",
                    f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("📉 Loss Curve")
        st.line_chart(df[['train/box_loss']])

        st.subheader("📈 Recall Curve")
        st.line_chart(df[['metrics/recall(B)']])

        cm_path = "analysis/confusion_matrix.csv"
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, index_col=0)
            fig = px.imshow(cm,
                            text_auto=True,
                            color_continuous_scale="Blues")
            st.subheader("📊 Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("results.csv not found.")

# ==========================================================
# FAILURE CASES
# ==========================================================
if page == "Failure Cases":
    st.title("🚨 Failure Cases")

    base = "analysis/failure_cases"

    case_type = st.selectbox(
        "Select Failure Type",
        ["False Positives", "False Negatives", "Small Objects"]
    )

    folder_map = {
        "False Positives": "false_positives",
        "False Negatives": "false_negatives",
        "Small Objects": "small_objects"
    }

    folder = os.path.join(base, folder_map[case_type])

    if os.path.exists(folder):
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".png"))]

        if images:
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 3].image(
                    os.path.join(folder, img),
                    caption=img,
                    use_container_width=True
                )
        else:
            st.info("No images found in this category.")
    else:
        st.warning("Failure folder not found.")
