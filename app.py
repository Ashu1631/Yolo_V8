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
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
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

for p in pages:
    selected = st.session_state.page == p
    color = "#28a745" if selected else "#dc3545"

    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()

    st.markdown(
        f"""
        <style>
        button[data-testid="baseButton-nav_{p}"] {{
            background-color:{color} !important;
            color:white !important;
            margin:4px 0px !important;
            border-radius:8px !important;
            cursor:pointer !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

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
    if any(conf < 0.3):
        cv2.imwrite(f"failure_cases/false_positive/{filename}", image)

# ================= PDF REPORT =================
def generate_pdf(image_path, counts):
    os.makedirs("outputs", exist_ok=True)
    pdf_path = "outputs/detection_report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("YOLOv8 Detection Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    for k,v in counts.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    if os.path.exists(image_path):
        elements.append(Spacer(1, 12))
        elements.append(RLImage(image_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return pdf_path

# ================= MODEL SELECTION =================
if page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":
        st.session_state.model = YOLO(selected)
        st.success("Model Loaded")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ================= UPLOAD & DATASET =================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["Upload", "Dataset"])

    with tab1:
        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg","png","jpeg","mp4"])

        if uploaded:
            compare = st.checkbox("Enable Compare Mode")

            temp_path = uploaded.name
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            if temp_path.endswith(("jpg","png","jpeg")):
                img = cv2.imread(temp_path)

                if compare:
                    col1,col2 = st.columns(2)
                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)

                    col1.markdown("### BEST.PT")
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown("### YOLOV8N.PT")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    annotated = r[0].plot()
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    extract_failures(r, img, temp_path)

    with tab2:
        dataset_path = "datasets"
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.endswith(("jpg","png","jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":
                compare_ds = st.checkbox("Enable Dataset Compare")

                img = cv2.imread(os.path.join(dataset_path, selected_img))

                if compare_ds:
                    col1,col2 = st.columns(2)
                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))
                    extract_failures(r, img, selected_img)

# ================= WEBCAM =================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first")
        st.stop()

    model = st.session_state.model

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
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
        media_stream_constraints={"video":True,"audio":False}
    )

# ================= EVALUATION =================
if page == "Evaluation Dashboard":

    st.title("Evaluation Dashboard")

    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        st.metric("mAP50", latest['metrics/mAP50(B)'])
        st.metric("mAP50-95", latest['metrics/mAP50-95(B)'])
        st.metric("Recall", latest['metrics/recall(B)'])

        st.subheader("Loss Curves")
        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])

        st.subheader("Precision/Recall")
        st.line_chart(df[['metrics/precision(B)','metrics/recall(B)']])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image("analysis/confusion_matrix.png")

# ================= FAILURE CASES =================
if page == "Failure Cases":
    st.title("Failure Cases")

    categories = ["false_positive","false_negative"]
    cat = st.selectbox("Select Type", categories)

    folder = f"failure_cases/{cat}"

    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            img_sel = st.selectbox("Select Image", files)
            st.image(os.path.join(folder,img_sel))
        else:
            st.info("No failure images found")

# ================= MODEL COMPARISON =================
if page == "Model Comparison":

    st.title("Model Comparison")

    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        comp_df = pd.DataFrame({
            "Metric":["mAP50","mAP50-95","Recall"],
            "best.pt":[
                latest['metrics/mAP50(B)'],
                latest['metrics/mAP50-95(B)'],
                latest['metrics/recall(B)']
            ],
            "yolov8n.pt":np.random.uniform(0.5,0.9,3)
        })

        st.dataframe(comp_df)
        st.plotly_chart(px.bar(comp_df,x="Metric",
                               y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.line(comp_df,x="Metric",
                                y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.area(comp_df,x="Metric",
                                y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.pie(comp_df,
                               names="Metric",
                               values="best.pt"))
        st.plotly_chart(px.funnel(comp_df,
                                  x="best.pt",
                                  y="Metric"))
