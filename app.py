import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ==================================================
# SIDEBAR STYLE (HAND + ACTIVE BORDER)
# ==================================================
st.markdown("""
<style>
.sidebar-btn button {
    width: 100%;
    border-radius: 8px;
    padding: 8px;
    cursor: pointer;
}
.sidebar-active button {
    border: 2px solid #00FFFF !important;
    background-color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION INIT
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "model_object" not in st.session_state:
    st.session_state.model_object = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

# ==================================================
# PASSWORD HASH
# ==================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ==================================================
# LOAD USERS
# ==================================================
with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k.strip(): v['password'].strip()
             for k, v in users_yaml['users'].items()}

# ==================================================
# LOGIN
# ==================================================
def login():
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users:
            stored = users[username]
            if stored == password or stored == hash_password(password):
                st.session_state.logged_in = True
                st.rerun()
        st.error("Invalid Credentials ❌")

if not st.session_state.logged_in:
    login()
    st.stop()

# ==================================================
# SIDEBAR NAVIGATION (BUTTON BASED)
# ==================================================
st.sidebar.markdown("## 🚀 Navigation")

nav_options = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard"
]

for option in nav_options:
    container_class = "sidebar-active" if st.session_state.page == option else "sidebar-btn"
    with st.sidebar.container():
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        if st.button(option, key=f"nav_{option}", use_container_width=True):
            st.session_state.page = option
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state.page

# ==================================================
# MODEL SELECTION
# ==================================================
if page == "Model Selection":

    st.title("📦 Model Selection")

    model_files = [f for f in os.listdir() if f.endswith(".pt")]

    selected_model = st.selectbox(
        "Please select the model",
        ["-- Select Model --"] + model_files
    )

    if selected_model != "-- Select Model --":
        if st.session_state.selected_model != selected_model:
            with st.spinner(f"Loading {selected_model}..."):
                st.session_state.model_object = YOLO(selected_model)
                st.session_state.selected_model = selected_model
            st.session_state.page = "Upload & Detect"
            st.rerun()

# ==================================================
# DETECTION COUNT FUNCTION
# ==================================================
def show_detection_counts(results):
    boxes = results[0].boxes
    if boxes is not None and len(boxes.cls) > 0:
        classes = boxes.cls.cpu().numpy()
        names = results[0].names
        counts = {}
        for c in classes:
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        st.subheader("📊 Detection Counts")
        st.json(counts)

# ==================================================
# UPLOAD & DETECT
# ==================================================
if page == "Upload & Detect":

    st.title("📤 Upload & Detect")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    uploaded_file = st.file_uploader(
        "Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    temp_path = None

    if uploaded_file:
        temp_path = os.path.join(
            tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # IMAGE
        if uploaded_file.name.lower().endswith(("jpg", "png", "jpeg")):
            start = time.time()
            results = model(temp_path, conf=conf, iou=iou)
            img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            end = time.time()
            st.image(img)
            st.success(f"Detection Time: {round(end-start,2)} sec")
            show_detection_counts(results)

        # VIDEO
        if uploaded_file.name.lower().endswith("mp4"):
            cap = cv2.VideoCapture(temp_path)
            FRAME_WINDOW = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, conf=conf, iou=iou)
                annotated = cv2.cvtColor(
                    results[0].plot(), cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated)
            cap.release()

    # ---------------- DATASET SECTION ----------------
    st.subheader("📂 Dataset Images")

    dataset_path = "datasets"

    if os.path.exists(dataset_path):

        dataset_images = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if dataset_images:

            selected_dataset_img = st.selectbox(
                "Select Dataset Image",
                ["-- Select Image --"] + dataset_images
            )

            if selected_dataset_img != "-- Select Image --":

                img_path = os.path.join(dataset_path, selected_dataset_img)

                results = model(img_path, conf=conf, iou=iou)
                img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                st.image(img)
                show_detection_counts(results)

                # MODEL COMPARISON FOR DATASET
                compare_dataset = st.checkbox("Compare best.pt vs yolov8n.pt")

                if compare_dataset and os.path.exists("best.pt") and os.path.exists("yolov8n.pt"):

                    model_best = YOLO("best.pt")
                    model_nano = YOLO("yolov8n.pt")

                    res1 = model_best(img_path, conf=conf, iou=iou)
                    res2 = model_nano(img_path, conf=conf, iou=iou)

                    img1 = cv2.cvtColor(res1[0].plot(), cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(res2[0].plot(), cv2.COLOR_BGR2RGB)

                    col1, col2 = st.columns(2)
                    col1.image(img1, caption="best.pt")
                    col2.image(img2, caption="yolov8n.pt")

# ==================================================
# WEBCAM LIVE (STUN + TURN + FPS)
# ==================================================
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.prev_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.25, iou=0.45)
        annotated = results[0].plot()

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24")

if page == "Webcam Detection":

    st.title("📷 Live Webcam Detection")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    webrtc_streamer(
        key="yolo-live",
        video_processor_factory=lambda: YOLOVideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": "turn:openrelay.metered.ca:80",
                    "username": "openrelayproject",
                    "credential": "openrelayproject"
                }
            ]
        },
    )

# ==================================================
# EVALUATION DASHBOARD (UNCHANGED)
# ==================================================
if page == "Evaluation Dashboard":

    st.title("📊 Model Evaluation")

    analysis_path = "analysis"
    metrics_file = os.path.join(analysis_path, "results.csv")

    if not os.path.exists(metrics_file):
        st.error("results.csv not found in analysis folder.")
        st.stop()

    df = pd.read_csv(metrics_file)

    precision_col = "metrics/precision(B)"
    recall_col = "metrics/recall(B)"
    map50_col = "metrics/mAP50(B)"
    map5095_col = "metrics/mAP50-95(B)"
    loss_col = "train/box_loss"

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("mAP50", f"{latest[map50_col]*100:.2f}%")
    col2.metric("mAP50-95", f"{latest[map5095_col]*100:.2f}%")
    col3.metric("Precision", f"{latest[precision_col]*100:.2f}%")
    col4.metric("Recall", f"{latest[recall_col]*100:.2f}%")

    st.line_chart(df[[loss_col]].rename(columns={loss_col: "Box Loss"}))
    st.line_chart(df[[precision_col]].rename(columns={precision_col: "Precision"}))
    st.line_chart(df[[recall_col]].rename(columns={recall_col: "Recall"}))

    for img in ["confusion_matrix.png",
                "PR_curve.png",
                "F1_curve.png",
                "results.png"]:
        path = os.path.join(analysis_path, img)
        if os.path.exists(path):
            st.image(path)
