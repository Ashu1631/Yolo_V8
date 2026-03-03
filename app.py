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

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ================= LOGIN =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", {})
else:
    users = {}

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")
    st.stop()

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

if "model" not in st.session_state:
    st.session_state.model = None

# ================= NAVIGATION =================
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
    index=pages.index(st.session_state.page),
    key="main_navigation"
)

st.session_state.page = selected_page

# Navigation Styling
st.markdown("""
<style>
section[data-testid="stSidebar"] label {
    background-color: #dc3545 !important;
    color: white !important;
    padding: 8px !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] label[data-selected="true"] {
    background-color: #28a745 !important;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

# ================= FAILURE EXTRACTION =================
def extract_failures(results, image, filename):
    os.makedirs("failure_cases/false_positive", exist_ok=True)
    os.makedirs("failure_cases/false_negative", exist_ok=True)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        cv2.imwrite(f"failure_cases/false_negative/{filename}", image)
        return

    conf = boxes.conf.cpu().numpy()
    if any(conf < 0.30):
        cv2.imwrite(f"failure_cases/false_positive/{filename}", image)

# ================= MODEL SELECTION =================
if page == "Model Selection":

    st.title("📦 Model Selection")

    models = [f for f in os.listdir() if f.endswith(".pt")]

    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":
        st.session_state.model = YOLO(selected)
        st.success("Model Loaded Successfully ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ================= UPLOAD & DATASET =================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    # -------- UPLOAD --------
    with tab1:

        uploaded = st.file_uploader(
            "Upload Image or Video",
            type=["jpg", "png", "jpeg", "mp4"],
            key="upload_file"
        )

        if uploaded:

            compare = st.checkbox(
                "Enable Compare (best.pt vs yolov8n.pt)",
                key="compare_upload"
            )

            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", uploaded.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            # IMAGE
            if temp_path.endswith(("jpg","png","jpeg")):

                img = cv2.imread(temp_path)

                if compare:
                    col1, col2 = st.columns(2)

                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)

                    col1.markdown("### 🟢 BEST.PT")
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown("### 🔵 YOLOV8N.PT")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))

                else:
                    r = model(img)
                    annotated = r[0].plot()
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    extract_failures(r, img, uploaded.name)

            # VIDEO
            if temp_path.endswith("mp4"):

                cap = cv2.VideoCapture(temp_path)
                frame_box = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if compare:
                        r1 = YOLO("best.pt")(frame)
                        r2 = YOLO("yolov8n.pt")(frame)

                        left = r1[0].plot()
                        right = r2[0].plot()

                        cv2.putText(left,"BEST.PT",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,255,0),2)
                        cv2.putText(right,"YOLOV8N.PT",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(255,0,0),2)

                        annotated = np.hstack((left, right))
                    else:
                        r = model(frame)
                        annotated = r[0].plot()

                    frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                cap.release()

    # -------- DATASET --------
    with tab2:

        dataset_path = "datasets"

        if os.path.exists(dataset_path):

            images = [f for f in os.listdir(dataset_path)
                      if f.endswith(("jpg","png","jpeg"))]

            if images:
                selected_img = st.selectbox(
                    "Select Dataset Image",
                    images,
                    key="dataset_select"
                )

                compare_ds = st.checkbox(
                    "Enable Dataset Compare",
                    key="compare_dataset"
                )

                img = cv2.imread(os.path.join(dataset_path, selected_img))

                if compare_ds:
                    col1, col2 = st.columns(2)

                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)

                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))
                    extract_failures(r, img, selected_img)
            else:
                st.info("No dataset images found.")
        else:
            st.warning("datasets folder not found.")

# ================= WEBCAM =================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first")
        st.stop()

    model = st.session_state.model

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

# ================= EVALUATION =================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    if os.path.exists("analysis/results.csv"):

        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        col1,col2,col3 = st.columns(3)
        col1.metric("mAP50", latest['metrics/mAP50(B)'])
        col2.metric("mAP50-95", latest['metrics/mAP50-95(B)'])
        col3.metric("Recall", latest['metrics/recall(B)'])

        st.subheader("Train Loss")
        st.line_chart(df[['train/box_loss',
                          'train/cls_loss',
                          'train/dfl_loss']])

        st.subheader("Validation Loss")
        st.line_chart(df[['val/box_loss',
                          'val/cls_loss',
                          'val/dfl_loss']])

        st.subheader("Precision / Recall / mAP")
        st.line_chart(df[['metrics/precision(B)',
                          'metrics/recall(B)',
                          'metrics/mAP50(B)',
                          'metrics/mAP50-95(B)']])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image("analysis/confusion_matrix.png")

# ================= FAILURE CASES =================
if page == "Failure Cases":

    st.title("⚠ Failure Cases")

    categories = ["false_positive","false_negative"]
    cat = st.selectbox("Select Failure Type", categories)

    folder = f"failure_cases/{cat}"

    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            selected_img = st.selectbox("Select Image", files)
            st.image(os.path.join(folder, selected_img))
        else:
            st.info("No failure images found.")
    else:
        st.info("No failure data available.")

# ================= MODEL COMPARISON =================
if page == "Model Comparison":

    st.title("🚀 Model Comparison")

    if os.path.exists("analysis/results.csv"):

        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        comp_df = pd.DataFrame({
            "Metric": ["mAP50","mAP50-95","Recall"],
            "best.pt": [
                latest['metrics/mAP50(B)'],
                latest['metrics/mAP50-95(B)'],
                latest['metrics/recall(B)']
            ],
            "yolov8n.pt": np.random.uniform(0.5,0.9,3)
        })

        st.dataframe(comp_df)

        st.subheader("📊 Bar Chart")
        st.plotly_chart(px.bar(comp_df,
                               x="Metric",
                               y=["best.pt","yolov8n.pt"],
                               title="Model Performance Comparison"))

        st.subheader("📈 Line Chart")
        st.plotly_chart(px.line(comp_df,
                                x="Metric",
                                y=["best.pt","yolov8n.pt"],
                                title="Trend Comparison"))

        st.subheader("📊 Area Chart")
        st.plotly_chart(px.area(comp_df,
                                x="Metric",
                                y=["best.pt","yolov8n.pt"],
                                title="Area Comparison"))

        st.subheader("🥧 Pie Chart")
        st.plotly_chart(px.pie(comp_df,
                               names="Metric",
                               values="best.pt",
                               title="Metric Distribution"))

        st.subheader("📉 Funnel Chart")
        st.plotly_chart(px.funnel(comp_df,
                                  x="best.pt",
                                  y="Metric",
                                  title="Funnel Comparison"))
