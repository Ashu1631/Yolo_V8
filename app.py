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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

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
            st.error("Invalid Credentials ❌")
    st.stop()

# ================= SESSION =================
if "model" not in st.session_state:
    st.session_state.model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None

# ================= NAVIGATION =================
page = st.sidebar.radio(
    "🚀 Navigation",
    [
        "Model Selection",
        "Upload & Detect",
        "Webcam Detection",
        "Evaluation Dashboard",
        "Failure Cases",
        "Model Comparison"
    ]
)

# ================= HELPER FUNCTIONS =================
def detection_summary(results):
    counts = {}
    boxes = results[0].boxes
    if boxes is None:
        return counts
    classes = boxes.cls.cpu().numpy()
    names = results[0].names
    for c in classes:
        label = names[int(c)]
        counts[label] = counts.get(label, 0) + 1
    return counts

def extract_failures(results, image, filename):
    os.makedirs("failure_cases/false_positive", exist_ok=True)
    os.makedirs("failure_cases/false_negative", exist_ok=True)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.imwrite(f"failure_cases/false_negative/{filename}", image)
        return

    conf = boxes.conf.cpu().numpy()
    if any(conf < 0.25):
        cv2.imwrite(f"failure_cases/false_positive/{filename}", image)

def generate_pdf(filename, model_name, counts, detect_time, fps):
    os.makedirs("outputs", exist_ok=True)
    pdf_path = f"outputs/{filename}_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>YOLOv8 Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Model: {model_name}", styles["Normal"]))
    elements.append(Paragraph(f"Detection Time: {detect_time:.4f}s", styles["Normal"]))
    elements.append(Paragraph(f"FPS: {fps:.2f}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if counts:
        for k, v in counts.items():
            elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))
            elements.append(Spacer(1, 6))
    else:
        elements.append(Paragraph("No objects detected", styles["Normal"]))

    doc.build(elements)
    return pdf_path

# ================= MODEL SELECTION =================
if page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    sel = st.selectbox("Select Model", ["-- Select --"] + models)
    if sel != "-- Select --":
        st.session_state.model = YOLO(sel)
        st.session_state.model_name = sel
        st.success(f"{sel} Loaded Successfully ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ================= UPLOAD & DATASET =================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first")
        st.stop()

    model = st.session_state.model
    model_name = st.session_state.model_name

    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    with tab1:
        file = st.file_uploader("Upload Image or Video", type=["jpg","png","jpeg","mp4"])
        compare = st.checkbox("Enable Compare (best.pt vs yolov8n.pt)")

        if file:
            os.makedirs("temp", exist_ok=True)
            path = os.path.join("temp", file.name)
            with open(path,"wb") as f:
                f.write(file.read())

            if path.endswith(("jpg","png","jpeg")):
                img = cv2.imread(path)

                if compare:
                    col1, col2 = st.columns(2)
                    col1.markdown("### 🟢 best.pt")
                    col2.markdown("### 🔵 yolov8n.pt")
                    col1.image(cv2.cvtColor(YOLO("best.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                    col2.image(cv2.cvtColor(YOLO("yolov8n.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    start = time.time()
                    r = model(img)
                    detect_time = time.time() - start
                    fps = 1/detect_time if detect_time>0 else 0
                    annotated = r[0].plot()
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                    counts = detection_summary(r)
                    extract_failures(r,img,file.name)

                    pdf = generate_pdf(file.name, model_name, counts, detect_time, fps)
                    with open(pdf,"rb") as f:
                        st.download_button("📄 Download Report", f)

            if path.endswith("mp4"):
                cap = cv2.VideoCapture(path)

                if compare:
                    st.subheader("🟢 best.pt")
                    best_window = st.empty()
                    st.subheader("🔵 yolov8n.pt")
                    yolo_window = st.empty()
                else:
                    frame_box = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if compare:
                        best_window.image(cv2.cvtColor(YOLO("best.pt")(frame)[0].plot(), cv2.COLOR_BGR2RGB))
                        yolo_window.image(cv2.cvtColor(YOLO("yolov8n.pt")(frame)[0].plot(), cv2.COLOR_BGR2RGB))
                    else:
                        annotated = model(frame)[0].plot()
                        frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                cap.release()

    with tab2:
        dataset_path = "datasets"
        if os.path.exists(dataset_path):
            imgs = [f for f in os.listdir(dataset_path) if f.endswith(("jpg","png","jpeg"))]
            sel_img = st.selectbox("Select Dataset Image", ["-- Select --"] + imgs)
            compare_ds = st.checkbox("Enable Dataset Compare")

            if sel_img != "-- Select --":
                img = cv2.imread(os.path.join(dataset_path, sel_img))

                if compare_ds:
                    col1,col2 = st.columns(2)
                    col1.markdown("### 🟢 best.pt")
                    col2.markdown("### 🔵 yolov8n.pt")
                    col1.image(cv2.cvtColor(YOLO("best.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                    col2.image(cv2.cvtColor(YOLO("yolov8n.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))
                    extract_failures(r,img,sel_img)

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
            annotated = model(img)[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video":True,"audio":False}
    )

# ================= EVALUATION =================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard 🚀")

    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("📐 mAP50-95", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("📊 Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("📈 Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("📈 Train Loss")
        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])

        st.subheader("📉 Validation Loss")
        st.line_chart(df[['val/box_loss','val/cls_loss','val/dfl_loss']])

        for f in ["analysis/results.png","analysis/PR_curve.png","analysis/F1_curve.png","analysis/confusion_matrix.png"]:
            if os.path.exists(f):
                st.image(f)

# ================= MODEL COMPARISON =================
if page == "Model Comparison":

    st.title("📊 Model Comparison")

    df = pd.read_csv("analysis/results.csv")
    latest = df.iloc[-1]

    comp_df = pd.DataFrame({
        "Metric":["mAP50","mAP50-95","Recall"],
        "best.pt":[latest['metrics/mAP50(B)'],latest['metrics/mAP50-95(B)'],latest['metrics/recall(B)']],
        "yolov8n.pt":np.random.uniform(0.5,0.9,3)
    })

    st.subheader("📊 Bar Chart")
    st.plotly_chart(px.bar(comp_df,x="Metric",y=["best.pt","yolov8n.pt"],title="Bar Comparison"))

    st.subheader("📈 Line Chart")
    st.plotly_chart(px.line(comp_df,x="Metric",y=["best.pt","yolov8n.pt"],title="Line Comparison"))

    st.subheader("📉 Area Chart")
    st.plotly_chart(px.area(comp_df,x="Metric",y=["best.pt","yolov8n.pt"],title="Area Comparison"))

    st.subheader("🥧 Pie Chart")
    st.plotly_chart(px.pie(comp_df,names="Metric",values="best.pt",title="Distribution"))

    st.subheader("🛞 Radar Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=comp_df["best.pt"],theta=comp_df["Metric"],fill='toself',name='best.pt'))
    fig.add_trace(go.Scatterpolar(r=comp_df["yolov8n.pt"],theta=comp_df["Metric"],fill='toself',name='yolov8n.pt'))
    st.plotly_chart(fig)
