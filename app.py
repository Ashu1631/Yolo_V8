import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard",
                   page_icon="🚀",
                   layout="wide")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# SESSION
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model_object" not in st.session_state:
    st.session_state.model_object = None

# ==================================================
# LOGIN
# ==================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k.strip(): v['password'].strip()
             for k, v in users_yaml['users'].items()}

if not st.session_state.logged_in:
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

    st.stop()

# ==================================================
# NAVIGATION (CLEAR ACTIVE)
# ==================================================
st.sidebar.markdown("## 🚀 Navigation")

pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases"
]

selected = st.sidebar.radio(
    "",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected
page = selected

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_counts(results):
    boxes = results[0].boxes
    counts = {}
    if boxes is not None and len(boxes.cls) > 0:
        classes = boxes.cls.cpu().numpy()
        names = results[0].names
        for c in classes:
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1
    return counts

def show_bar_chart(counts):
    if counts:
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Detected Objects Per Class")
        st.pyplot(fig)

# ==================================================
# MODEL SELECTION
# ==================================================
if page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected_model = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected_model != "-- Select --":
        st.session_state.model_object = YOLO(selected_model)
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==================================================
# UPLOAD & DATASET
# ==================================================
if page == "Upload & Detect":

    if not st.session_state.model_object:
        st.warning("Please load model first.")
        st.stop()

    model = st.session_state.model_object

    uploaded = st.file_uploader("Upload Image/Video",
                                type=["jpg", "png", "jpeg", "mp4"])

    # ================= IMAGE =================
    if uploaded and uploaded.name.lower().endswith(("jpg", "png", "jpeg")):
        path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())

        results = model(path)
        img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        st.image(img)

        counts = get_counts(results)
        st.json(counts)
        show_bar_chart(counts)

        # Compare
        if st.checkbox("Compare best.pt vs yolov8n.pt"):
            model1 = YOLO("best.pt")
            model2 = YOLO("yolov8n.pt")

            r1 = model1(path)
            r2 = model2(path)

            img1 = cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            col1.image(img1, caption="best.pt")
            col2.image(img2, caption="yolov8n.pt")

    # ================= VIDEO =================
    if uploaded and uploaded.name.lower().endswith("mp4"):
        path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated = results[0].plot()
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        cap.release()

    # ================= DATASET =================
    st.subheader("📂 Dataset Images")

    dataset_path = "datasets"

    if not uploaded and os.path.exists(dataset_path):
        dataset_images = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if dataset_images:
            selected_img = st.selectbox(
                "Select Dataset Image",
                ["-- Select Image --"] + dataset_images
            )

            if selected_img != "-- Select Image --":
                img_path = os.path.join(dataset_path, selected_img)
                results = model(img_path)
                img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(img)

                if st.checkbox("Compare Models (Dataset)"):
                    model1 = YOLO("best.pt")
                    model2 = YOLO("yolov8n.pt")

                    r1 = model1(img_path)
                    r2 = model2(img_path)

                    img1 = cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB)

                    col1, col2 = st.columns(2)
                    col1.image(img1, caption="best.pt")
                    col2.image(img2, caption="yolov8n.pt")

# ==================================================
# WEBCAM (SAFE VERSION)
# ==================================================
class SafeProcessor(VideoProcessorBase):
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
        video_processor_factory=SafeProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ==================================================
# EVALUATION DASHBOARD
# ==================================================
if page == "Evaluation Dashboard":
    path = "analysis/results.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col3.metric("Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.line_chart(df[['train/box_loss']])
        st.line_chart(df[['metrics/precision(B)']])
        st.line_chart(df[['metrics/recall(B)']])

        cm_path = "analysis/confusion_matrix.csv"
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, index_col=0)
            fig = px.imshow(cm, text_auto=True,
                            color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("results.csv not found.")

# ==================================================
# FAILURE CASES
# ==================================================
if page == "Failure Cases":
    st.title("🚨 Failure Case Analysis")

    base = "analysis/failure_cases"
    case = st.selectbox("Select Type",
                        ["False Positives", "False Negatives", "Small Objects"])

    folder_map = {
        "False Positives": "false_positives",
        "False Negatives": "false_negatives",
        "Small Objects": "small_objects"
    }

    folder = os.path.join(base, folder_map[case])

    if os.path.exists(folder):
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".png"))]

        if images:
            cols = st.columns(3)
            for i, img_name in enumerate(images):
                img_path = os.path.join(folder, img_name)
                cols[i % 3].image(img_path,
                                  caption=img_name,
                                  use_container_width=True)
        else:
            st.info("No images found.")
    else:
        st.info("Failure folder not found.")
