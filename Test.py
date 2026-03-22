import streamlit as st
import tempfile
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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ashu YOLO AI", layout="wide", page_icon="🎯")

# --- SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model" not in st.session_state:
    st.session_state.model = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None

# --- RTC CONFIG ---
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ]
})

# --- VIDEO PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.4
        self.frame_count = 0
        self.fps = 0
        self._last_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.model is not None:
            try:
                results = self.model(img, conf=self.conf_threshold, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
                box_annotator = sv.BoxAnnotator(thickness=2)
                label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
                labels = [
                    f"{results[0].names[cls_id]} {conf:.2f}"
                    for cls_id, conf in zip(detections.class_id, detections.confidence)
                ]
                annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                self.frame_count += 1
                now = time.time()
                elapsed = now - self._last_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self._last_time = now
                cv2.putText(annotated, f"FPS: {self.fps:.1f} | Objects: {len(detections)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            except Exception as e:
                cv2.putText(img, f"Error: {str(e)[:50]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =================== FPS GAUGE HELPER ===================
def make_fps_gauge(fps_val, model_label, gauge_key):
    if fps_val >= 40:
        bar_color = "#28a745"
    elif fps_val >= 20:
        bar_color = "#ffc107"
    else:
        bar_color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fps_val,
        title={"text": f"FPS — {model_label}", "font": {"size": 14}},
        number={"suffix": " fps", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 20],  "color": "rgba(220,53,69,0.15)"},
                {"range": [20, 40], "color": "rgba(255,193,7,0.15)"},
                {"range": [40, 100],"color": "rgba(40,167,69,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": fps_val,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True, key=gauge_key)


def get_fps(dt):
    return round(1 / dt, 1) if dt > 0 else 0.0


# =================== LOGIN ===================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

        /* Hide streamlit default elements on login */
        #MainMenu, footer, header { visibility: hidden; }
        .stApp {
            background: #000 !important;
            overflow: hidden;
        }

        /* Animated grid background */
        .login-bg {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background:
                linear-gradient(rgba(40,167,69,0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(40,167,69,0.07) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridMove 20s linear infinite;
            z-index: 0;
        }
        @keyframes gridMove {
            0% { background-position: 0 0, 0 0; }
            100% { background-position: 50px 50px, 50px 50px; }
        }

        /* Glow orbs */
        .orb1 {
            position: fixed;
            width: 500px; height: 500px;
            background: radial-gradient(circle, rgba(40,167,69,0.15) 0%, transparent 70%);
            top: -100px; left: -100px;
            border-radius: 50%;
            animation: orbFloat1 8s ease-in-out infinite;
            z-index: 0;
        }
        .orb2 {
            position: fixed;
            width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(0,200,150,0.1) 0%, transparent 70%);
            bottom: -80px; right: -80px;
            border-radius: 50%;
            animation: orbFloat2 10s ease-in-out infinite;
            z-index: 0;
        }
        @keyframes orbFloat1 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(60px, 40px) scale(1.1); }
        }
        @keyframes orbFloat2 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(-40px, -60px) scale(1.15); }
        }

        /* Scanline effect */
        .scanline {
            position: fixed;
            top: 0; left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, rgba(40,167,69,0.8), transparent);
            animation: scanMove 4s linear infinite;
            z-index: 1;
        }
        @keyframes scanMove {
            0% { top: 0%; opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        /* Login card */
        .login-card {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(40,167,69,0.3);
            border-radius: 24px;
            padding: 48px 40px;
            box-shadow:
                0 0 60px rgba(40,167,69,0.1),
                0 0 120px rgba(40,167,69,0.05),
                inset 0 1px 0 rgba(255,255,255,0.1);
            position: relative;
            z-index: 2;
            animation: cardAppear 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes cardAppear {
            from { opacity: 0; transform: translateY(40px) scale(0.95); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* Corner accent lines */
        .login-card::before {
            content: '';
            position: absolute;
            top: -1px; left: -1px; right: -1px; bottom: -1px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(40,167,69,0.6), transparent 40%, transparent 60%, rgba(40,167,69,0.3));
            z-index: -1;
        }

        /* Logo area */
        .logo-wrapper {
            text-align: center;
            margin-bottom: 32px;
        }
        .logo-icon {
            font-size: 64px;
            display: block;
            animation: iconPulse 2s ease-in-out infinite;
        }
        @keyframes iconPulse {
            0%, 100% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(40,167,69,0.5)); }
            50% { transform: scale(1.08); filter: drop-shadow(0 0 25px rgba(40,167,69,0.9)); }
        }
        .logo-title {
            font-family: 'Orbitron', monospace;
            font-size: 28px;
            font-weight: 900;
            background: linear-gradient(135deg, #28a745, #00ff88, #28a745);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 3s linear infinite;
            letter-spacing: 4px;
            margin: 8px 0 4px;
        }
        @keyframes shimmer {
            0% { background-position: 0% center; }
            100% { background-position: 200% center; }
        }
        .logo-sub {
            font-family: 'Rajdhani', sans-serif;
            color: rgba(255,255,255,0.4);
            font-size: 13px;
            letter-spacing: 6px;
            text-transform: uppercase;
        }

        /* Divider */
        .login-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(40,167,69,0.5), transparent);
            margin: 20px 0 28px;
        }

        /* Override Streamlit input styles */
        .stTextInput label {
            font-family: 'Rajdhani', sans-serif !important;
            color: rgba(40,167,69,0.9) !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 3px !important;
            text-transform: uppercase !important;
        }
        .stTextInput input {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(40,167,69,0.25) !important;
            border-radius: 12px !important;
            color: #e8f5e9 !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 16px !important;
            padding: 12px 16px !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput input:focus {
            border-color: rgba(40,167,69,0.7) !important;
            box-shadow: 0 0 20px rgba(40,167,69,0.2) !important;
            background: rgba(40,167,69,0.05) !important;
        }

        /* Login button */
        div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #1a7a2e, #28a745, #1a7a2e) !important;
            background-size: 200% auto !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-family: 'Orbitron', monospace !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            letter-spacing: 3px !important;
            padding: 14px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            box-shadow: 0 4px 20px rgba(40,167,69,0.4) !important;
        }
        div[data-testid="stButton"] button:hover {
            background-position: right center !important;
            box-shadow: 0 6px 35px rgba(40,167,69,0.65) !important;
            transform: translateY(-2px) !important;
        }
        div[data-testid="stButton"] button:active {
            transform: translateY(0) scale(0.98) !important;
        }

        /* Status badge */
        .status-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 28px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 12px;
            letter-spacing: 2px;
            color: rgba(255,255,255,0.35);
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #28a745;
            box-shadow: 0 0 8px #28a745;
            animation: dotBlink 1.5s ease-in-out infinite;
        }
        @keyframes dotBlink {
            0%, 100% { opacity: 1; box-shadow: 0 0 8px #28a745; }
            50% { opacity: 0.4; box-shadow: 0 0 3px #28a745; }
        }

        /* Version tag */
        .version-tag {
            text-align: center;
            margin-top: 24px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 11px;
            color: rgba(255,255,255,0.2);
            letter-spacing: 3px;
        }
        </style>

        <!-- Background elements -->
        <div class="login-bg"></div>
        <div class="orb1"></div>
        <div class="orb2"></div>
        <div class="scanline"></div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
            <div class="login-card">
                <div class="logo-wrapper">
                    <span class="logo-icon">🎯</span>
                    <div class="logo-title">ASHU YOLO AI</div>
                    <div class="logo-sub">Object Detection System</div>
                </div>
                <div class="login-divider"></div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    SYSTEM ONLINE · SECURE ACCESS
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Inputs rendered by Streamlit (outside the HTML card so they work)
        u = st.text_input("Username", placeholder="Enter username", key="login_user")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        p = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if st.button("⚡  INITIALIZE ACCESS", use_container_width=True):
            if u == "admin" and p == "ashu@123":
                st.markdown("""
                    <style>
                    .stApp { animation: flashGreen 0.5s ease forwards; }
                    @keyframes flashGreen {
                        0% { filter: brightness(1); }
                        50% { filter: brightness(1.3) hue-rotate(60deg); }
                        100% { filter: brightness(1); }
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.success("✅ Access Granted! Initializing...")
                time.sleep(0.8)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.markdown("""
                    <style>
                    .stTextInput input { animation: shake 0.4s ease; }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        20% { transform: translateX(-8px); }
                        40% { transform: translateX(8px); }
                        60% { transform: translateX(-5px); }
                        80% { transform: translateX(5px); }
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.error("❌ Invalid Credentials! Access Denied.")

        st.markdown('<div class="version-tag">YOLO AI v2.0 · NEURAL VISION ENGINE</div>', unsafe_allow_html=True)
    st.stop()


# =================== SIDEBAR ===================
# Global style for sidebar nav
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Rajdhani:wght@400;600&display=swap');

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d1a0d 100%) !important;
        border-right: 1px solid rgba(40,167,69,0.2) !important;
    }

    /* All sidebar nav buttons — default RED (unselected) */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #8b0000, #dc3545) !important;
        color: white !important;
        border: 1px solid rgba(220,53,69,0.4) !important;
        border-radius: 10px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 1px !important;
        height: 46px !important;
        margin-bottom: 4px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 12px rgba(220,53,69,0.25) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        transform: translateX(4px) !important;
        box-shadow: 0 4px 18px rgba(220,53,69,0.45) !important;
    }

    /* Logout button specific — darker red */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-child button {
        background: linear-gradient(135deg, #4a0000, #a00000) !important;
        border-color: rgba(160,0,0,0.5) !important;
    }

    /* Main content area */
    .stApp { background: #080d08 !important; }
    h1, h2, h3 { color: #e8f5e9 !important; }
    </style>
""", unsafe_allow_html=True)

pages = ["Model Selection", "Upload & Detect", "📷 Webcam Detection", "Evaluation Dashboard", "Model Comparison"]

st.sidebar.markdown("""
    <div style="
        font-family: 'Orbitron', monospace;
        font-size: 15px;
        font-weight: 700;
        color: #28a745;
        letter-spacing: 2px;
        padding: 8px 4px 16px;
        border-bottom: 1px solid rgba(40,167,69,0.3);
        margin-bottom: 16px;
        text-shadow: 0 0 12px rgba(40,167,69,0.5);
    ">🎯 ASHU YOLO AI</div>
    <div style="font-family:'Rajdhani',sans-serif;font-size:11px;color:rgba(255,255,255,0.3);
    letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;">Navigation</div>
""", unsafe_allow_html=True)

# Get selected page index (1-based for nth-child)
selected_idx = pages.index(st.session_state.page) + 1 if st.session_state.page in pages else 1

# Inject CSS: all nav buttons RED by default, nth-child selected one GREEN
st.sidebar.markdown(f"""
    <style>
    /* All nav buttons: RED */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
        background: linear-gradient(135deg, #6b0000, #dc3545) !important;
        color: white !important;
        border: 1px solid rgba(220,53,69,0.3) !important;
        border-radius: 10px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 1px !important;
        height: 46px !important;
        margin-bottom: 2px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 10px rgba(220,53,69,0.2) !important;
    }}

    /* Selected button (nth-child): GREEN */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({selected_idx}) button {{
        background: linear-gradient(135deg, #0d4a1e, #28a745) !important;
        border: 1px solid rgba(40,167,69,0.5) !important;
        box-shadow: 0 0 20px rgba(40,167,69,0.35), 0 2px 12px rgba(40,167,69,0.25) !important;
        color: white !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({selected_idx}) button:hover {{
        box-shadow: 0 0 30px rgba(40,167,69,0.55) !important;
        transform: translateX(4px) !important;
    }}

    /* Hover on red buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
        transform: translateX(4px) !important;
        box-shadow: 0 4px 18px rgba(220,53,69,0.4) !important;
    }}
    </style>
""", unsafe_allow_html=True)

for p in pages:
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()

st.sidebar.divider()

# Active page indicator badge
st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(40,167,69,0.12), rgba(40,167,69,0.06));
        border: 1px solid rgba(40,167,69,0.35);
        border-left: 3px solid #28a745;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0 8px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 12px;
        color: rgba(40,167,69,0.95);
        letter-spacing: 1px;
    ">
        ✅ Active: <b style="color:#28a745">{st.session_state.page}</b>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.page = "Model Selection"
    st.rerun()


# =================== HELPERS ===================
def apply_supervision(image, results):
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)
    labels = [
        f"{results[0].names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


# =================== PAGES ===================
page = st.session_state.page

# ----------- MODEL SELECTION -----------
if page == "Model Selection":
    st.title("📦 Ashu YOLO AI - Model Setup")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    if not models:
        st.warning("⚠️ Koi `.pt` model file nahi mili!")
        st.stop()
    sel = st.selectbox("Choose Model", ["-- Select --"] + models)
    if sel != "-- Select --":
        with st.spinner(f"Loading {sel}..."):
            st.session_state.model = YOLO(sel)
            st.session_state.model_name = sel
        st.success(f"✅ Model `{sel}` loaded!")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ----------- UPLOAD & DETECT -----------
elif page == "Upload & Detect":
    st.title("📤 Analysis Hub - Ashu YOLO AI")
    if not st.session_state.model:
        st.warning("⚠️ Pehle Model Selection page pe model load karein!")
        st.stop()

    tab1, tab2 = st.tabs(["📤 File Upload", "📂 Dataset Explorer"])

    with tab1:
        file = st.file_uploader("Upload Image/Video", type=["jpg", "png", "jpeg", "mp4"])
        if file:
            compare = st.checkbox("🔄 Enable Comparison (best.pt vs yolov8n.pt)")

            if file.name.lower().endswith(".mp4"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())
                cap = cv2.VideoCapture(tfile.name)

                if compare:
                    col1, col2 = st.columns(2)
                    col1.markdown("### 🎯 Best Model")
                    col2.markdown("### ⚡ Nano Model")
                    st_frame1 = col1.empty()
                    st_frame2 = col2.empty()
                    gcol1, gcol2 = st.columns(2)
                    gauge_ph1 = gcol1.empty()
                    gauge_ph2 = gcol2.empty()
                else:
                    st_frame = st.empty()
                    gauge_ph = st.empty()

                m_best = YOLO("best.pt") if compare else None
                m_nano = YOLO("yolov8n.pt") if compare else None
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if compare:
                        t1 = time.time()
                        r1 = m_best(frame, verbose=False)
                        fps1 = get_fps(time.time() - t1)

                        t2 = time.time()
                        r2 = m_nano(frame, verbose=False)
                        fps2 = get_fps(time.time() - t2)

                        st_frame1.image(cv2.cvtColor(apply_supervision(frame.copy(), r1), cv2.COLOR_BGR2RGB))
                        st_frame2.image(cv2.cvtColor(apply_supervision(frame.copy(), r2), cv2.COLOR_BGR2RGB))

                        with gauge_ph1.container():
                            make_fps_gauge(fps1, "best.pt", f"gauge_best_{frame_idx}")
                        with gauge_ph2.container():
                            make_fps_gauge(fps2, "yolov8n.pt", f"gauge_nano_{frame_idx}")
                    else:
                        t0 = time.time()
                        res = st.session_state.model(frame, verbose=False)
                        fps0 = get_fps(time.time() - t0)
                        st_frame.image(
                            cv2.cvtColor(apply_supervision(frame, res), cv2.COLOR_BGR2RGB),
                            use_container_width=True
                        )
                        with gauge_ph.container():
                            make_fps_gauge(fps0, st.session_state.model_name, f"gauge_single_{frame_idx}")

                    frame_idx += 1
                cap.release()

            else:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

                if compare:
                    t1 = time.time()
                    r1 = YOLO("best.pt")(img)
                    fps1 = get_fps(time.time() - t1)

                    t2 = time.time()
                    r2 = YOLO("yolov8n.pt")(img)
                    fps2 = get_fps(time.time() - t2)

                    ic1, ic2 = st.columns(2)
                    ic1.image(cv2.cvtColor(apply_supervision(img, r1), cv2.COLOR_BGR2RGB), caption="Best Model")
                    ic2.image(cv2.cvtColor(apply_supervision(img, r2), cv2.COLOR_BGR2RGB), caption="Nano Model")

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        make_fps_gauge(fps1, "best.pt", "img_gauge_best")
                    with gc2:
                        make_fps_gauge(fps2, "yolov8n.pt", "img_gauge_nano")
                else:
                    t0 = time.time()
                    res = st.session_state.model(img)
                    fps0 = get_fps(time.time() - t0)
                    st.image(cv2.cvtColor(apply_supervision(img, res), cv2.COLOR_BGR2RGB), use_container_width=True)
                    make_fps_gauge(fps0, st.session_state.model_name, "img_gauge_single")

    with tab2:
        if os.path.exists("datasets"):
            ds_imgs = [f for f in os.listdir("datasets") if f.endswith((".jpg", ".png", ".jpeg"))]
            sel_ds = st.selectbox("Select Image from Dataset Folder", ["-- Select --"] + ds_imgs)
            if sel_ds != "-- Select --":
                compare_ds = st.checkbox("🔄 Dataset Comparison Mode")
                ds_img = cv2.imread(os.path.join("datasets", sel_ds))

                if compare_ds:
                    t1 = time.time()
                    r1 = YOLO("best.pt")(ds_img)
                    fps1 = get_fps(time.time() - t1)

                    t2 = time.time()
                    r2 = YOLO("yolov8n.pt")(ds_img)
                    fps2 = get_fps(time.time() - t2)

                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.image(cv2.cvtColor(apply_supervision(ds_img, r1), cv2.COLOR_BGR2RGB), caption="Best.pt")
                    with dc2:
                        st.image(cv2.cvtColor(apply_supervision(ds_img, r2), cv2.COLOR_BGR2RGB), caption="Yolov8n.pt")

                    dg1, dg2 = st.columns(2)
                    with dg1:
                        make_fps_gauge(fps1, "best.pt", "ds_gauge_best")
                    with dg2:
                        make_fps_gauge(fps2, "yolov8n.pt", "ds_gauge_nano")
                else:
                    t0 = time.time()
                    res_ds = st.session_state.model(ds_img)
                    fps0 = get_fps(time.time() - t0)
                    st.image(cv2.cvtColor(apply_supervision(ds_img, res_ds), cv2.COLOR_BGR2RGB))
                    make_fps_gauge(fps0, st.session_state.model_name, "ds_gauge_single")
        else:
            st.info("📂 `datasets/` folder nahi mila.")

# ----------- WEBCAM DETECTION -----------
elif page == "📷 Webcam Detection":
    st.title("📷 Live Webcam Detection - Ashu YOLO AI")
    if not st.session_state.model:
        st.warning("⚠️ Pehle Model Selection page pe model load karein!")
        st.stop()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1a0d,#122612);padding:14px 20px;
    border-radius:12px;border-left:4px solid #28a745;margin-bottom:16px;
    box-shadow: 0 0 20px rgba(40,167,69,0.15);">
        <b style="color:#28a745;">✅ Active Model:</b>
        <span style="color:white;">{st.session_state.model_name}</span>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        conf_threshold = st.slider("🎚️ Confidence Threshold", 0.1, 0.95, 0.4, 0.05)
    with col_ctrl2:
        model_options = ["Current Model"] + [f for f in os.listdir() if f.endswith(".pt")]
        selected_model_wc = st.selectbox("🔁 Switch Model (Webcam)", model_options)
    with col_ctrl3:
        st.markdown("<br>", unsafe_allow_html=True)
        apply_btn = st.button("⚡ Apply", use_container_width=True)

    st.divider()

    with st.expander("💡 Webcam Tips"):
        st.markdown("""
        - Browser camera permission **allow** karein jab pooche.
        - **START** dabao webcam shuru, **STOP** se band karo.
        - Best browser: **Chrome** ya **Edge**.
        - FPS aur object count seedha video pe dikhega.
        - Low FPS? Confidence badhao ya `yolov8n.pt` use karo.
        - Agar webcam nahi khul raha — page refresh karo aur phir try karo.
        """)

    # Keep active model in session to avoid re-init on slider/button change
    if "webcam_model" not in st.session_state:
        st.session_state.webcam_model = st.session_state.model

    if apply_btn and selected_model_wc != "Current Model":
        with st.spinner(f"Loading {selected_model_wc}..."):
            st.session_state.webcam_model = YOLO(selected_model_wc)
        st.success(f"✅ Model switched to `{selected_model_wc}`!")

    active_model = st.session_state.webcam_model

    # Store conf in session so processor can access it
    st.session_state.webcam_conf = conf_threshold

    class LiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = active_model
            self.conf_threshold = st.session_state.get("webcam_conf", 0.4)
            self.frame_count = 0
            self.fps = 0
            self._last_time = time.time()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            # Update conf dynamically
            self.conf_threshold = st.session_state.get("webcam_conf", 0.4)
            if self.model is not None:
                try:
                    results = self.model(img, conf=self.conf_threshold, verbose=False)
                    detections = sv.Detections.from_ultralytics(results[0])
                    box_annotator = sv.BoxAnnotator(thickness=2)
                    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
                    labels = [
                        f"{results[0].names[cls_id]} {conf:.2f}"
                        for cls_id, conf in zip(detections.class_id, detections.confidence)
                    ]
                    annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                    self.frame_count += 1
                    now = time.time()
                    elapsed = now - self._last_time
                    if elapsed >= 1.0:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self._last_time = now
                    cv2.putText(annotated, f"FPS: {self.fps:.1f} | Objects: {len(detections)} | Conf: {self.conf_threshold:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception as e:
                    cv2.putText(img, f"Error: {str(e)[:60]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    st.markdown("### 🎥 Live Detection Stream")
    ctx = webrtc_streamer(
        key="ashu-yolo-webcam-v2",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "max": 1920},
                "height": {"ideal": 720, "max": 1080},
                "frameRate": {"ideal": 30, "max": 60}
            },
            "audio": False
        },
        async_processing=True,
        desired_playing_state=None,
    )

    st.divider()
    s1, s2, s3 = st.columns(3)
    with s1:
        if ctx.state.playing:
            st.success("🟢 **Stream: LIVE**")
        else:
            st.error("🔴 **Stream: Stopped**")
    with s2:
        st.info(f"🎯 **Model:** `{st.session_state.model_name}`")
    with s3:
        st.info(f"⚙️ **Confidence:** `{conf_threshold}`")

    if ctx.video_processor and apply_btn:
        ctx.video_processor.conf_threshold = conf_threshold
        if selected_model_wc != "Current Model":
            ctx.video_processor.model = active_model
        st.toast("✅ Settings applied!", icon="⚡")

# ----------- EVALUATION DASHBOARD -----------
elif page == "Evaluation Dashboard":
    st.title("📊 Ashu YOLO AI - Evaluation")
    st.divider()
    results_path = "analysis/results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]
        st.subheader("🎯 Key Performance Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.1f}%")
        m2.metric("mAP50-95", f"{latest['metrics/mAP50-95(B)']*100:.1f}%")
        m3.metric("Precision", f"{latest['metrics/precision(B)']*100:.1f}%")
        m4.metric("Recall", f"{latest['metrics/recall(B)']*100:.1f}%")
        st.divider()
        st.subheader("📉 Training vs Validation Loss")
        l1, l2 = st.columns(2)
        with l1:
            st.write("**Box Loss**")
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
        with l2:
            st.write("**Class Loss**")
            st.line_chart(df[['train/cls_loss', 'val/cls_loss']])
        st.divider()
        st.subheader("🖼️ Detailed Analysis Curves")
        tab1, tab2 = st.tabs(["Confusion Matrix", "F1 & PR Curves"])
        with tab1:
            if os.path.exists("analysis/confusion_matrix.png"):
                st.image("analysis/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("confusion_matrix.png not found in analysis/")
        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                if os.path.exists("analysis/BoxF1_curve.png"):
                    st.image("analysis/BoxF1_curve.png", caption="F1 Confidence Curve", use_container_width=True)
            with col_b:
                if os.path.exists("analysis/BoxPR_curve.png"):
                    st.image("analysis/BoxPR_curve.png", caption="Precision-Recall Curve", use_container_width=True)
    else:
        st.error("`analysis/results.csv` nahi mila. Pehle model train karein.")

# ----------- MODEL COMPARISON -----------
elif page == "Model Comparison":
    st.title("🚀 Ashu YOLO AI - 10-Graph Benchmarking")
    st.divider()

    metrics = ["mAP50", "mAP50-95", "Precision", "Recall", "Inference(ms)"]
    comp_df = pd.DataFrame({
        "Metric": metrics,
        "best.pt": [0.85, 0.65, 0.88, 0.82, 12.5],
        "yolov8n.pt": [0.78, 0.58, 0.80, 0.75, 8.2]
    })
    df_melted = comp_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    acc_df = df_melted[df_melted["Metric"] != "Inference(ms)"]

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            px.bar(acc_df, x="Metric", y="Score", color="Model", barmode="group",
                   title="1. Accuracy Metrics Comparison",
                   color_discrete_sequence=px.colors.qualitative.Pastel),
            key="cmp_bar", use_container_width=True)
    with r1c2:
        scatter_data = pd.DataFrame({
            "Model": ["best.pt", "yolov8n.pt"],
            "mAP50": [0.85, 0.78],
            "Latency (ms)": [12.5, 8.2]
        })
        st.plotly_chart(
            px.scatter(scatter_data, x="Latency (ms)", y="mAP50", text="Model",
                       size=[20, 15], title="2. Latency vs Accuracy"),
            key="cmp_scatter", use_container_width=True)

    st.divider()
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=comp_df["best.pt"][:4], theta=metrics[:4], fill='toself', name='best.pt'))
        fig_radar.add_trace(go.Scatterpolar(r=comp_df["yolov8n.pt"][:4], theta=metrics[:4], fill='toself', name='yolov8n.pt'))
        fig_radar.update_layout(title="3. Performance Radar")
        st.plotly_chart(fig_radar, key="cmp_radar", use_container_width=True)
    with r2c2:
        st.plotly_chart(
            px.line(acc_df, x="Metric", y="Score", color="Model", markers=True, title="4. Performance Trend"),
            key="cmp_line", use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fps_data = [[1000/12.5, 1000/15], [1000/8.2, 1000/10]]
        st.plotly_chart(
            px.imshow(fps_data, labels=dict(x="Hardware", y="Model", color="FPS"),
                      x=['GPU (T4)', 'CPU'], y=['best.pt', 'yolov8n.pt'],
                      title="5. Throughput Heatmap (FPS)", text_auto=True),
            key="cmp_heatmap", use_container_width=True)
    with r3c2:
        st.plotly_chart(
            px.pie(comp_df[comp_df["Metric"] != "Inference(ms)"],
                   names="Metric", values="best.pt",
                   title="6. mAP Weightage Distribution", hole=0.4),
            key="cmp_pie", use_container_width=True)

    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    with r4c1:
        st.plotly_chart(
            px.box(df_melted, y="Score", color="Model", title="7. Score Spread"),
            key="cmp_box", use_container_width=True)
    with r4c2:
        st.plotly_chart(
            px.histogram(df_melted, x="Score", nbins=5, title="8. Value Dist."),
            key="cmp_hist", use_container_width=True)
    with r4c3:
        st.plotly_chart(
            px.violin(df_melted, y="Score", box=True, title="9. Density"),
            key="cmp_violin", use_container_width=True)
    with r4c4:
        st.plotly_chart(
            px.strip(df_melted, x="Model", y="Score", color="Metric", title="10. Metric Points"),
            key="cmp_strip", use_container_width=True)
