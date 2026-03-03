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
import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard",
                   page_icon="🚀",
                   layout="wide")

# ==========================================================
# SESSION
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
# NAVIGATION (Green Selected / Red Unselected)
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
            <div style="background:#28a745;padding:10px;border-radius:8px;
                        color:white;font-weight:bold;margin-bottom:6px;">
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
def save_image(image, prefix="output"):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/{prefix}_{timestamp}.jpg"
    cv2.imwrite(path, image)
    return path

def generate_pdf_report(image_path, counts, fps=None):
    os.makedirs("outputs", exist_ok=True)
    pdf_path = "outputs/detection_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("YOLOv8 Detection Report", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Detected Objects:", styles['Normal']))
    for k, v in counts.items():
        elements.append(Paragraph(f"{k}: {v}", styles['Normal']))

    if fps:
        elements.append(Paragraph(f"FPS: {fps:.2f}", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(RLImage(image_path, width=4*inch, height=3*inch))

    doc.build(elements)
    return pdf_path

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

    # IMAGE & VIDEO UPLOAD
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

                fps = 1/(end-start)
                annotated = results[0].plot()
                img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(img)
                counts = get_counts(results)
                st.json(counts)
                st.success(f"FPS: {fps:.2f}")

                saved = save_image(annotated, "image")
                st.download_button("⬇ Download Image",
                                   open(saved,"rb"),
                                   file_name=os.path.basename(saved))

                pdf = generate_pdf_report(saved, counts, fps)
                st.download_button("⬇ Download PDF Report",
                                   open(pdf,"rb"),
                                   file_name="detection_report.pdf")

            # VIDEO
            if uploaded.name.lower().endswith("mp4"):
                cap = cv2.VideoCapture(path)
                frame_window = st.empty()

                start = time.time()
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame)
                    annotated = results[0].plot()
                    frame_window.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    frame_count += 1

                cap.release()
                end = time.time()
                fps = frame_count/(end-start)
                st.success(f"Video FPS: {fps:.2f}")

    # DATASET
    with tab2:
        dataset_path = "datasets"
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.lower().endswith((".jpg",".png",".jpeg"))]
            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"]+images)
            if selected_img != "-- Select --":
                img_path = os.path.join(dataset_path, selected_img)
                results = model(img_path)
                annotated = results[0].plot()
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

# ==========================================================
# WEBCAM (STABLE)
# ==========================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model

    class WebcamProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.prev_time = time.time()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            annotated = results[0].plot()

            current = time.time()
            fps = 1/(current-self.prev_time)
            self.prev_time = current
            cv2.putText(annotated,
                        f"FPS: {fps:.2f}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,255,0),2)

            return av.VideoFrame.from_ndarray(annotated,
                                              format="bgr24")

    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=WebcamProcessor,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )

# ==========================================================
# EVALUATION DASHBOARD
# ==========================================================
if page == "Evaluation Dashboard":
    csv_path = "analysis/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        latest = df.iloc[-1]

        col1,col2,col3 = st.columns(3)
        col1.metric("mAP50",
                    f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("Precision",
                    f"{latest['metrics/precision(B)']*100:.2f}%")
        col3.metric("Recall",
                    f"{latest['metrics/recall(B)']*100:.2f}%")

        st.line_chart(df[['train/box_loss']])
        st.line_chart(df[['metrics/recall(B)']])
        st.line_chart(df[['metrics/precision(B)']])

        cm_img = "analysis/confusion_matrix.png"
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix")

# ==========================================================
# MODEL PERFORMANCE COMPARISON
# ==========================================================
if page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    csv_path = "analysis/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        fig = px.line(df,
                      y=["metrics/mAP50(B)",
                         "metrics/precision(B)",
                         "metrics/recall(B)"],
                      title="Model Metrics Over Epochs")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FAILURE CASES (AUTO EXTRACTION READY)
# ==========================================================
if page == "Failure Cases":
    st.title("🚨 Failure Cases")
    base = "analysis/failure_cases"
    case_type = st.selectbox("Select Type",
                             ["false_positives",
                              "false_negatives",
                              "small_objects"])
    folder = os.path.join(base, case_type)
    if os.path.exists(folder):
        images = os.listdir(folder)
        cols = st.columns(3)
        for i,img in enumerate(images):
            cols[i%3].image(os.path.join(folder,img),
                            use_container_width=True)
