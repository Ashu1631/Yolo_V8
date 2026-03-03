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
import time
import datetime
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
# LOGIN
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
            f"""
            <div style="background:#28a745;padding:10px;
                        border-radius:8px;color:white;
                        font-weight:bold;margin-bottom:6px;">
                👉 {p}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page

# ==========================================================
# HELPERS
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
    tab1, tab2 = st.tabs(["📤 Upload File", "📂 Dataset Folder"])

    # ---------------- UPLOAD ----------------
    with tab1:
        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg", "png", "jpeg", "mp4"])

        if uploaded:
            path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.read())

            # IMAGE
            if uploaded.name.lower().endswith(("jpg", "png", "jpeg")):
                start = time.time()
                results = model(path)
                end = time.time()

                fps = 1 / (end - start)
                annotated = results[0].plot()

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                st.success(f"FPS: {fps:.2f}")
                st.json(get_counts(results))

            # VIDEO
            if uploaded.name.lower().endswith("mp4"):

                cap = cv2.VideoCapture(path)

                os.makedirs("outputs", exist_ok=True)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_original = cap.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path = "outputs/processed_video.mp4"
                out = cv2.VideoWriter(output_path, fourcc,
                                      fps_original, (width, height))

                frame_window = st.empty()
                fps_list = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    start = time.time()
                    results = model(frame)
                    annotated = results[0].plot()
                    end = time.time()

                    fps = 1 / (end - start)
                    fps_list.append(fps)

                    cv2.putText(annotated,
                                f"FPS: {fps:.2f}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0), 2)

                    out.write(annotated)

                    frame_window.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    )

                cap.release()
                out.release()

                st.success("Video Saved Automatically ✅")

                with open(output_path, "rb") as file:
                    st.download_button(
                        "⬇ Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

                st.subheader("📈 FPS Graph")
                st.line_chart(pd.DataFrame({"FPS": fps_list}))
                st.info(f"Average FPS: {np.mean(fps_list):.2f}")

    # ---------------- DATASET ----------------
    with tab2:
        dataset_path = "datasets"

        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":
                img_path = os.path.join(dataset_path, selected_img)
                results = model(img_path)
                annotated = results[0].plot()
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

# ==========================================================
# WEBCAM
# ==========================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model

    class WebcamProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.prev = time.time()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            annotated = results[0].plot()

            now = time.time()
            fps = 1 / (now - self.prev)
            self.prev = now

            cv2.putText(annotated,
                        f"FPS: {fps:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(annotated,
                                              format="bgr24")

    webrtc_streamer(
        key="webcam_key",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==========================================================
# EVALUATION
# ==========================================================
if page == "Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard 🚀")

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

        st.line_chart(df[['train/box_loss']])
        st.line_chart(df[['metrics/precision(B)']])
        st.line_chart(df[['metrics/recall(B)']])

        cm_path = "analysis/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")

# ==========================================================
# FAILURE CASES
# ==========================================================
if page == "Failure Cases":
    st.title("🚨 Failure Cases")

    base = "analysis/failure_cases"
    case = st.selectbox("Select Type",
                        ["false_positives",
                         "false_negatives",
                         "small_objects"])

    folder = os.path.join(base, case)

    if os.path.exists(folder):
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith((".jpg", ".png"))]

        if len(images) == 0:
            st.warning("No Data Found ⚠")
        else:
            cols = st.columns(3)
            for i, img in enumerate(images):
                cols[i % 3].image(
                    os.path.join(folder, img),
                    use_container_width=True
                )
    else:
        st.warning("No Data Found ⚠")

# ==========================================================
# MODEL COMPARISON
# ==========================================================
if page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    csv_path = "analysis/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]

        metrics = {
            "mAP50": latest['metrics/mAP50(B)'],
            "Precision": latest['metrics/precision(B)'],
            "Recall": latest['metrics/recall(B)']
        }

        st.line_chart(df[['metrics/mAP50(B)',
                          'metrics/precision(B)',
                          'metrics/recall(B)']])

        st.plotly_chart(px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title="Bar Chart"
        ))

        st.plotly_chart(px.pie(
            names=list(metrics.keys()),
            values=list(metrics.values()),
            title="Pie Chart"
        ))
