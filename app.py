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

# PDF Report Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# --- Page Config ---
st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide", page_icon="🚀")

# --- Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Dummy User logic (users.yaml se read karega agar hai toh)
users = {"admin": {"password": "123"}} # Default
if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", users)

if not st.session_state.logged_in:
    st.title("🔐 Enterprise Login")
    col_l, col_r = st.columns([1, 2])
    with col_l:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if u in users and users[u]["password"] == p:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials ❌")
    st.stop()

# --- Navigation Logic ---
pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases",
    "Model Comparison"
]

if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model" not in st.session_state:
    st.session_state.model = None

# --- Sidebar Navigation with Custom CSS ---
st.sidebar.markdown("## 🚀 Navigation")

for p in pages:
    selected = st.session_state.page == p
    color = "#28a745" if selected else "#dc3545"
    cursor_style = "pointer" if selected else "default"

    # Injecting CSS for Button Styling
    st.sidebar.markdown(f"""
        <style>
        button[key="nav_{p}"] {{
            background-color: {color} !important;
            color: white !important;
            border-radius: 6px !important;
            cursor: {cursor_style} !important;
            border: none !important;
            height: 45px;
        }}
        </style>
    """, unsafe_allow_html=True)

    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()

current_page = st.session_state.page

# --- Helper Functions ---
def detection_summary(results):
    counts = {}
    boxes = results[0].boxes
    if boxes is not None:
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
    else:
        conf = boxes.conf.cpu().numpy()
        if any(conf < 0.25):
            cv2.imwrite(f"failure_cases/false_positive/{filename}", image)

def generate_pdf_report(filename, model_name, counts, detect_time, fps):
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
    if counts:
        for k, v in counts.items():
            elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    doc.build(elements)
    return pdf_path

# ================= PAGES LOGIC =================

if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    if not models:
        st.info("No .pt models found in directory.")
    else:
        sel = st.selectbox("Select Model to Load", ["-- Select --"] + models)
        if sel != "-- Select --":
            with st.spinner("Loading Model..."):
                st.session_state.model = YOLO(sel)
                st.session_state.model_name = sel
                st.session_state.page = "Upload & Detect"
                st.rerun()

elif current_page == "Upload & Detect":
    if not st.session_state.model:
        st.warning("⚠️ Pehle Model Selection page se model load karein!")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    with tab1:
        file = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])
        if file:
            compare = st.checkbox("🔄 Enable Comparison Mode (best.pt vs yolov8n.pt)")
            os.makedirs("temp", exist_ok=True)
            path = os.path.join("temp", file.name)
            with open(path, "wb") as f: f.write(file.read())

            if path.endswith(("jpg","png","jpeg")):
                img = cv2.imread(path)
                if compare:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🟢 BEST.PT")
                        r1 = YOLO("best.pt")(img)
                        st.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))
                    with c2:
                        st.markdown("### 🔵 YOLOV8N.PT")
                        r2 = YOLO("yolov8n.pt")(img)
                        st.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    start = time.time()
                    r = model(img)
                    dt = time.time() - start
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))
                    # Report logic
                    pdf = generate_pdf_report(file.name, st.session_state.model_name, detection_summary(r), dt, 1/dt)
                    with open(pdf, "rb") as f:
                        st.download_button("📄 Download Report", f, file_name=f"{file.name}_report.pdf")

            elif path.endswith("mp4"):
                cap = cv2.VideoCapture(path)
                frame_box = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    if compare:
                        r1, r2 = YOLO("best.pt")(frame), YOLO("yolov8n.pt")(frame)
                        combined = np.hstack((r1[0].plot(), r2[0].plot()))
                        frame_box.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                    else:
                        r = model(frame)
                        frame_box.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))
                cap.release()

    with tab2:
        if os.path.exists("datasets"):
            imgs = [f for f in os.listdir("datasets") if f.endswith(("jpg","png","jpeg"))]
            sel_img = st.selectbox("Select Dataset Image", ["-- Select --"] + imgs)
            if sel_img != "-- Select --":
                compare_ds = st.checkbox("🔄 Compare in Dataset")
                img = cv2.imread(os.path.join("datasets", sel_img))
                if compare_ds:
                    c1, c2 = st.columns(2)
                    with c1: 
                        st.markdown("### 🟢 BEST.PT")
                        st.image(cv2.cvtColor(YOLO("best.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                    with c2: 
                        st.markdown("### 🔵 YOLOV8N.PT")
                        st.image(cv2.cvtColor(YOLO("yolov8n.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    st.image(cv2.cvtColor(model(img)[0].plot(), cv2.COLOR_BGR2RGB))

elif current_page == "Evaluation Dashboard":
    st.title("📊 Model Evaluation Dashboard")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP50 🎯", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        m2.metric("mAP50-95 📈", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        m3.metric("Precision ✅", f"{latest['metrics/precision(B)']*100:.2f}%")
        m4.metric("Recall 🔍", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.divider()
        st.markdown("### 📉 Stacked Loss Curves (Train vs Val)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("📦 Box Loss")
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
        with c2:
            st.subheader("🏷️ Class Loss")
            st.line_chart(df[['train/cls_loss', 'val/cls_loss']])
        with c3:
            st.subheader("🌀 DFL Loss")
            st.line_chart(df[['train/dfl_loss', 'val/dfl_loss']])

        st.divider()
        if os.path.exists("analysis/confusion_matrix.png"):
            st.markdown("### 🧩 Confusion Matrix")
            st.image("analysis/confusion_matrix.png", use_container_width=True)
    else:
        st.error("analysis/results.csv not found!")

elif current_page == "Failure Cases":
    st.title("⚠️ Failure Analysis")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("False Positives (Low Conf)")
        path = "failure_cases/false_positive"
        if os.path.exists(path):
            imgs = os.listdir(path)
            for i in imgs: st.image(os.path.join(path, i), caption=i)
    with cols[1]:
        st.subheader("False Negatives (No Detect)")
        path = "failure_cases/false_negative"
        if os.path.exists(path):
            imgs = os.listdir(path)
            for i in imgs: st.image(os.path.join(path, i), caption=i)

elif current_page == "Model Comparison":
    st.title("🚀 Comparison Dashboard")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]
        
        comp_data = {
            "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
            "best.pt": [latest['metrics/mAP50(B)'], latest['metrics/mAP50-95(B)'], latest['metrics/precision(B)'], latest['metrics/recall(B)']],
            "yolov8n.pt": [0.82, 0.65, 0.78, 0.75] # Dummy comparison
        }
        comp_df = pd.DataFrame(comp_data)
        st.table(comp_df)

        # Wagon Wheel / Radar Chart
        st.subheader("🛞 Radar Chart (Wagon Wheel)")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=comp_df["best.pt"], theta=comp_df["Metric"], fill='toself', name='best.pt'))
        fig.add_trace(go.Scatterpolar(r=comp_df["yolov8n.pt"], theta=comp_df["Metric"], fill='toself', name='yolov8n.pt'))
        st.plotly_chart(fig)
    else:
        st.warning("No results to compare.")

elif current_page == "Webcam Detection":
    if not st.session_state.model:
        st.warning("Please load model first!")
    else:
        model = st.session_state.model
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                r = model(img)
                return av.VideoFrame.from_ndarray(r[0].plot(), format="bgr24")

        webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor, 
                        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}))
)
