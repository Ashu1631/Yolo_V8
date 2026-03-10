import streamlit as st
import os
import cv2
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import supervision as sv
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- Page Config ---
st.set_page_config(page_title="Ashu YOLO AI", layout="wide", page_icon="🎯")

# --- Session State ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None

# ================= LOGIN PAGE =================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        /* Pure page background */
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                        url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-position: center;
        }
        
        /* Login Container Fix */
        [data-testid="stVerticalBlock"] > div:has(div.login-box) {
            background: transparent !important;
        }

        .login-box {
            background: rgba(255, 255, 255, 0.08);
            padding: 2rem;
            border-radius: 20px;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 400px;
            margin: auto;
            text-align: center;
        }

        /* Input and Label Styling */
        label { color: #28a745 !important; font-weight: bold !important; margin-bottom: 5px; }
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 10px !important;
        }
        
        h2 { color: white !important; font-family: 'Segoe UI', sans-serif; margin-bottom: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    # Spacing to center vertically
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # centering using columns
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        # Wrap everything in a single custom div
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("<h2>🎯 ASHU YOLO AI</h2>", unsafe_allow_html=True)
        u = st.text_input("Username", placeholder="Enter admin username")
        p = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Access Dashboard", use_container_width=True):
            if u == "admin" and p == "ashu@123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invaild Credentials! ❌")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
# ================= SIDEBAR NAVIGATION =================
pages = ["Model Selection", "Upload & Detect", "Webcam Detection", "Evaluation Dashboard", "Failure Cases", "Model Comparison"]
st.sidebar.markdown("## 🚀 Navigation")
for p in pages:
    selected = st.session_state.page == p
    color = "#28a745" if selected else "#dc3545"
    st.sidebar.markdown(f"<style>button[key='nav_{p}'] {{ background-color: {color} !important; color: white !important; border-radius: 10px !important; cursor: pointer !important; height: 48px !important; }}</style>", unsafe_allow_html=True)
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()

# --- Supervision Helper ---
def apply_supervision(image, results):
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
    labels = [f"{results[0].names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

def get_fps_chart(dt):
    fps = 1/dt if dt > 0 else 0
    fig = go.Figure(go.Indicator(mode="gauge+number", value=fps, title={'text': "FPS Gauge"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#28a745"}}))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ================= PAGES LOGIC =================
page = st.session_state.page

if page == "Model Selection":
    st.title("📦 Ashu YOLO AI - Model Setup")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    sel = st.selectbox("Choose Model", ["-- Select --"] + models)
    if sel != "-- Select --":
        st.session_state.model = YOLO(sel)
        st.session_state.model_name = sel
        st.session_state.page = "Upload & Detect"; st.rerun()

# 2. UPLOAD & DETECT
elif page == "Upload & Detect":
    st.title("📤 Analysis Hub - Ashu YOLO AI")
    if not st.session_state.model: 
        st.warning("⚠️ Pehle model load karein!")
        st.stop()
    
    tab1, tab2 = st.tabs(["📤 File Upload", "📂 Dataset Explorer"])
    
    with tab1:
        file = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])
        if file:
            compare = st.checkbox("🔄 Enable Comparison (best.pt vs yolov8n.pt)")
            path = os.path.join("temp", file.name); os.makedirs("temp", exist_ok=True)
            with open(path, "wb") as f: f.write(file.read())
            
            if file.name.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(path)
                
                # FPS Calculation START (Dono modes ke liye common)
                start_time = time.time()
                res_current = st.session_state.model(img)
                dt = time.time() - start_time
                # FPS Calculation END
                
                if compare:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🟢 BEST.PT")
                        st.image(cv2.cvtColor(apply_supervision(img, YOLO("best.pt")(img)), cv2.COLOR_BGR2RGB))
                    with c2:
                        st.markdown("### 🔵 YOLOV8N.PT")
                        st.image(cv2.cvtColor(apply_supervision(img, YOLO("yolov8n.pt")(img)), cv2.COLOR_BGR2RGB))
                else:
                    st.image(cv2.cvtColor(apply_supervision(img, res_current), cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # FPS Graph ab comparison ke niche bhi dikhega
                st.plotly_chart(get_fps_chart(dt))
            
            # Video logic same rahegi...

    with tab2:
        if os.path.exists("datasets"):
            ds_imgs = [f for f in os.listdir("datasets") if f.endswith((".jpg", ".png", ".jpeg"))]
            sel_ds = st.selectbox("Select Image from Dataset Folder", ["-- Select --"] + ds_imgs)
            if sel_ds != "-- Select --":
                compare_ds = st.checkbox("🔄 Dataset Comparison Mode")
                ds_img = cv2.imread(os.path.join("datasets", sel_ds))
                
                # Dataset FPS Calculation
                start_ds = time.time()
                res_ds = st.session_state.model(ds_img)
                dt_ds = time.time() - start_ds

                if compare_ds:
                    cl1, cl2 = st.columns(2)
                    with cl1: st.image(cv2.cvtColor(apply_supervision(ds_img, YOLO("best.pt")(ds_img)), cv2.COLOR_BGR2RGB), caption="Best.pt")
                    with cl2: st.image(cv2.cvtColor(apply_supervision(ds_img, YOLO("yolov8n.pt")(ds_img)), cv2.COLOR_BGR2RGB), caption="Yolov8n.pt")
                else:
                    st.image(cv2.cvtColor(apply_supervision(ds_img, res_ds), cv2.COLOR_BGR2RGB))
                
                # Dataset FPS Graph
                st.plotly_chart(get_fps_chart(dt_ds))

elif page == "Webcam Detection":
    st.title("📷 Ashu YOLO AI - Live Stream")
    if not st.session_state.model: st.warning("Load model first!")
    else:
        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                res = st.session_state.model(img)
                return av.VideoFrame.from_ndarray(apply_supervision(img, res), format="bgr24")

        webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor, 
                        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))

elif page == "Evaluation Dashboard":
    st.title("📊 Ashu YOLO AI - Evaluation")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP50 🎯", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        m2.metric("mAP50-95 📈", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        st.divider()
        st.markdown("### 📉 Stacked Loss Curves")
        
        l1, l2 = st.columns(2)
        with l1: st.line_chart(df[['train/box_loss', 'val/box_loss']])
        with l2: st.line_chart(df[['train/cls_loss', 'val/cls_loss']])
        if os.path.exists("analysis/confusion_matrix.png"): st.image("analysis/confusion_matrix.png", use_container_width=True)
    else: st.error("Results file missing.")

elif page == "Model Comparison":
    st.title("🚀 Ashu YOLO AI - 10-Graph Dashboard")
    metrics = ["mAP50", "mAP50-95", "Precision", "Recall", "Inference(ms)"]
    comp_df = pd.DataFrame({"Metric": metrics, "best.pt": [0.85, 0.65, 0.88, 0.82, 12], "yolov8n.pt": [0.78, 0.58, 0.80, 0.75, 8]})
    
    g1, g2 = st.columns(2)
    with g1: st.plotly_chart(px.bar(comp_df, x="Metric", y=["best.pt", "yolov8n.pt"], barmode="group", title="1. Bar Comparison"))
    with g2: st.plotly_chart(px.line(comp_df, x="Metric", y=["best.pt", "yolov8n.pt"], markers=True, title="2. Metric Trend"))
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=comp_df["best.pt"][:4], theta=metrics[:4], fill='toself', name='best.pt'))
    fig_radar.add_trace(go.Scatterpolar(r=comp_df["yolov8n.pt"][:4], theta=metrics[:4], fill='toself', name='yolov8n.pt'))
    st.plotly_chart(fig_radar) # 3. Wagon Wheel Radar
    
    st.plotly_chart(px.pie(comp_df, names="Metric", values="best.pt", title="6. Pie Distribution"))
    st.plotly_chart(px.box(comp_df, y=["best.pt", "yolov8n.pt"], title="7. Box Plot"))
    st.plotly_chart(px.histogram(comp_df, x="best.pt", title="8. Histogram"))
    st.plotly_chart(px.violin(comp_df, y="best.pt", title="9. Violin Plot"))
    st.plotly_chart(px.strip(comp_df, x="Metric", y="best.pt", title="10. Strip Plot"))
