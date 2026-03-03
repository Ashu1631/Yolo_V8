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

users = {"admin": {"password": "123"}} 
if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", users)

if not st.session_state.logged_in:
    st.title("🔐 Enterprise Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")
    st.stop()

# --- Navigation ---
pages = ["Model Selection", "Upload & Detect", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None

st.sidebar.markdown("## 🚀 Navigation")
for p in pages:
    selected = st.session_state.page == p
    color = "#28a745" if selected else "#dc3545"
    cursor = "pointer" if selected else "default"
    st.sidebar.markdown(f"<style>button[key='nav_{p}'] {{ background-color: {color} !important; color: white !important; cursor: {cursor} !important; }}</style>", unsafe_allow_html=True)
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()

current_page = st.session_state.page

# --- Helper Functions ---
def detection_summary(results):
    counts = {}
    if results[0].boxes is not None:
        for c in results[0].boxes.cls.cpu().numpy():
            label = results[0].names[int(c)]
            counts[label] = counts.get(label, 0) + 1
    return counts

def generate_pdf_report(filename, model_name, counts, dt, fps):
    os.makedirs("outputs", exist_ok=True)
    pdf_path = f"outputs/{filename}_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = [Paragraph("<b>YOLOv8 Detection Report</b>", getSampleStyleSheet()["Title"]), Spacer(1, 12)]
    elements.append(Paragraph(f"Model: {model_name} | FPS: {fps:.2f}", getSampleStyleSheet()["Normal"]))
    for k, v in counts.items(): elements.append(Paragraph(f"{k}: {v}", getSampleStyleSheet()["Normal"]))
    doc.build(elements)
    return pdf_path

# ================= PAGES =================

if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    sel = st.selectbox("Select Model", ["-- Select --"] + models)
    if sel != "-- Select --":
        st.session_state.model = YOLO(sel)
        st.session_state.model_name = sel
        st.session_state.page = "Upload & Detect"
        st.rerun()

elif current_page == "Upload & Detect":
    if not st.session_state.model: st.warning("Load model first!"); st.stop()
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])
    with tab1:
        file = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])
        if file:
            compare = st.checkbox("🔄 Enable Comparison")
            path = os.path.join("temp", file.name); os.makedirs("temp", exist_ok=True)
            with open(path, "wb") as f: f.write(file.read())
            img = cv2.imread(path)
            if compare:
                c1, c2 = st.columns(2)
                with c1: st.image(cv2.cvtColor(YOLO("best.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB), caption="BEST.PT")
                with c2: st.image(cv2.cvtColor(YOLO("yolov8n.pt")(img)[0].plot(), cv2.COLOR_BGR2RGB), caption="YOLOV8N.PT")
            else:
                r = st.session_state.model(img)
                st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))

elif current_page == "Webcam Detection":
    st.title("📷 Real-time Detection")
    if not st.session_state.model:
        st.warning("Please load a model first!")
    else:
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = st.session_state.model(img)
                return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

        # FIX: Corrected bracket nesting
        webrtc_streamer(
            key="webcam",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False}
        )

elif current_page == "Evaluation Dashboard":
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP50 🎯", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        m2.metric("mAP50-95 📈", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        st.divider()
        st.markdown("### 📉 Stacked Loss Curves")
        st.line_chart(df[['train/box_loss', 'val/box_loss']])
        if os.path.exists("analysis/confusion_matrix.png"):
            st.image("analysis/confusion_matrix.png", caption="Confusion Matrix")
    else:
        st.error("Results file not found.")

elif current_page == "Model Comparison":
    st.title("🚀 Comparison Dashboard")
    # Radar chart logic...
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0.8, 0.7, 0.9], theta=['mAP', 'Precision', 'Recall'], fill='toself'))
    st.plotly_chart(fig)
