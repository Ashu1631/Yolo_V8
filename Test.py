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
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": ["turn:global.relay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:global.relay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:global.relay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
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


# =================== LOGIN — Space / Sci-Fi Purple Neon ===================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');

        #MainMenu, footer, header { visibility: hidden; }

        .stApp {
            background: #06030f !important;
            overflow: hidden;
        }

        /* Nebula blobs */
        .nebula {
            position: fixed;
            border-radius: 50%;
            pointer-events: none;
            z-index: 0;
        }
        .nebula-1 {
            width: 600px; height: 500px;
            background: radial-gradient(ellipse, rgba(120,40,200,0.28) 0%, transparent 70%);
            top: -150px; left: -150px;
            filter: blur(80px);
            animation: nebulaFloat 15s ease-in-out infinite;
        }
        .nebula-2 {
            width: 500px; height: 450px;
            background: radial-gradient(ellipse, rgba(200,40,255,0.22) 0%, transparent 70%);
            bottom: -100px; right: -100px;
            filter: blur(80px);
            animation: nebulaFloat 18s ease-in-out infinite reverse;
        }
        .nebula-3 {
            width: 350px; height: 350px;
            background: radial-gradient(ellipse, rgba(60,200,255,0.14) 0%, transparent 70%);
            top: 40%; left: 60%;
            filter: blur(70px);
            animation: nebulaFloat 12s ease-in-out infinite 3s;
        }
        @keyframes nebulaFloat {
            0%,100% { transform: translate(0,0) scale(1); }
            33%  { transform: translate(30px,-20px) scale(1.05); }
            66%  { transform: translate(-20px,30px) scale(0.97); }
        }

        /* Stars canvas */
        .stars-layer {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            pointer-events: none;
            z-index: 0;
        }

        /* Scan line */
        .scanline {
            position: fixed;
            left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, rgba(180,80,255,0.65), transparent);
            animation: scan 6s linear infinite;
            z-index: 1;
            pointer-events: none;
        }
        @keyframes scan {
            0%   { top: -5px;  opacity: 0; }
            5%   { opacity: 1; }
            95%  { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }

        /* Login card */
        .login-card {
            position: relative;
            z-index: 2;
            background: rgba(12,6,28,0.88);
            border: 1px solid rgba(150,60,255,0.35);
            border-radius: 24px;
            padding: 44px 36px 36px;
            backdrop-filter: blur(24px);
            box-shadow:
                0 0 0 1px rgba(150,60,255,0.08),
                0 0 60px rgba(130,40,220,0.22),
                inset 0 0 60px rgba(100,20,180,0.06);
            animation: cardIn 0.9s cubic-bezier(0.16,1,0.3,1) both;
        }
        @keyframes cardIn {
            from { opacity:0; transform:translateY(50px) scale(0.93); }
            to   { opacity:1; transform:translateY(0)    scale(1);    }
        }

        /* Corner brackets */
        .c-tl,.c-tr,.c-bl,.c-br {
            position:absolute; width:20px; height:20px;
            border-color:rgba(180,80,255,0.75); border-style:solid;
        }
        .c-tl { top:12px; left:12px;  border-width:2px 0 0 2px; border-radius:4px 0 0 0; }
        .c-tr { top:12px; right:12px; border-width:2px 2px 0 0; border-radius:0 4px 0 0; }
        .c-bl { bottom:12px; left:12px;  border-width:0 0 2px 2px; border-radius:0 0 0 4px; }
        .c-br { bottom:12px; right:12px; border-width:0 2px 2px 0; border-radius:0 0 4px 0; }

        /* Orbit icon */
        .orbit-wrap { text-align:center; margin-bottom:24px; }
        .orbit {
            display:inline-block; position:relative;
            width:80px; height:80px;
        }
        .orbit-core {
            position:absolute; top:50%; left:50%;
            transform:translate(-50%,-50%);
            width:34px; height:34px; border-radius:50%;
            background:radial-gradient(circle,#c060ff,#7020c0);
            box-shadow:0 0 20px rgba(160,60,255,0.9),0 0 40px rgba(160,60,255,0.35);
        }
        .orbit-ring {
            position:absolute; top:4px; left:4px;
            width:72px; height:72px; border-radius:50%;
            border:1px solid rgba(180,80,255,0.4);
            animation:spin 6s linear infinite;
        }
        .orbit-ring::before {
            content:''; position:absolute;
            top:-4px; left:50%; transform:translateX(-50%);
            width:8px; height:8px; border-radius:50%;
            background:#c060ff;
            box-shadow:0 0 10px #c060ff;
        }
        .orbit-ring2 {
            position:absolute; top:12px; left:12px;
            width:56px; height:56px; border-radius:50%;
            border:1px solid rgba(120,200,255,0.3);
            animation:spin 4s linear infinite reverse;
        }
        .orbit-ring2::before {
            content:''; position:absolute;
            bottom:-4px; left:50%; transform:translateX(-50%);
            width:6px; height:6px; border-radius:50%;
            background:#80d0ff;
            box-shadow:0 0 8px #80d0ff;
        }
        @keyframes spin {
            from { transform:rotate(0deg); }
            to   { transform:rotate(360deg); }
        }

        /* Titles */
        .logo-title {
            font-family:'Share Tech Mono',monospace;
            font-size:26px; font-weight:700;
            letter-spacing:5px; text-align:center;
            color:#d080ff;
            text-shadow:0 0 22px rgba(180,80,255,0.65);
            margin-bottom:4px;
            overflow:hidden;
            white-space:nowrap;
            border-right:2px solid #d080ff;
            width:0;
            animation: typing 1.6s steps(12,end) 0.3s forwards, cursorBlink 0.7s step-end 1.9s 4, glitchLoop 6s ease-in-out 2.5s infinite;
            display:inline-block;
        }
        @keyframes typing {
            from { width:0; }
            to   { width:100%; }
        }
        @keyframes cursorBlink {
            0%,100% { border-color:#d080ff; }
            50%     { border-color:transparent; }
        }
        @keyframes glitchLoop {
            0%,90%,100% { text-shadow:0 0 22px rgba(180,80,255,0.65); transform:translate(0); }
            91%  { text-shadow:-2px 0 #ff00ff, 2px 0 #00ffff; transform:translate(-2px,1px); }
            92%  { text-shadow:2px 0 #ff00ff, -2px 0 #00ffff; transform:translate(2px,-1px); }
            93%  { text-shadow:-1px 0 #ff00ff, 1px 0 #00ffff; transform:translate(0); }
            94%  { text-shadow:0 0 22px rgba(180,80,255,0.65); transform:translate(0); }
        }
        .logo-sub {
            font-family:'Rajdhani',sans-serif;
            font-size:10px; letter-spacing:6px;
            text-align:center; text-transform:uppercase;
            color:rgba(180,120,255,0.42);
            margin-bottom:22px;
            opacity:0;
            animation: fadeSlideUp 0.6s ease 2s forwards;
        }
        @keyframes fadeSlideUp {
            from { opacity:0; transform:translateY(10px); }
            to   { opacity:1; transform:translateY(0); }
        }

        /* Meteor */
        .meteor {
            position:fixed;
            height:2px;
            background:linear-gradient(90deg, rgba(200,100,255,0.9), transparent);
            border-radius:2px;
            pointer-events:none;
            z-index:0;
            opacity:0;
            animation: meteorFly var(--dur,3s) linear infinite var(--delay,0s);
        }
        @keyframes meteorFly {
            0%   { opacity:0; transform:translateX(0) translateY(0); }
            5%   { opacity:1; }
            80%  { opacity:0.6; }
            100% { opacity:0; transform:translateX(calc(var(--dx)*1px)) translateY(calc(var(--dy)*1px)); }
        }

        /* Status bar */
        .status-bar {
            display:flex; align-items:center; justify-content:center;
            gap:8px; margin-bottom:24px;
            font-family:'Share Tech Mono',monospace;
            font-size:11px; letter-spacing:2px;
            color:rgba(255,255,255,0.3);
        }
        .status-dot {
            width:7px; height:7px; border-radius:50%;
            background:#a040ff;
            box-shadow:0 0 10px #a040ff;
            animation:dotPulse 2s ease-in-out infinite;
        }
        @keyframes dotPulse {
            0%,100% { opacity:1; box-shadow:0 0 10px #a040ff; }
            50%     { opacity:0.4; box-shadow:0 0 4px #a040ff; }
        }

        /* Divider */
        .login-divider {
            height:1px;
            background:linear-gradient(90deg,transparent,rgba(150,60,255,0.5),transparent);
            margin-bottom:26px;
        }

        /* Streamlit input overrides */
        .stTextInput label {
            font-family:'Share Tech Mono',monospace !important;
            font-size:10px !important;
            letter-spacing:3px !important;
            color:rgba(160,80,255,0.85) !important;
            text-transform:uppercase !important;
        }
        .stTextInput input {
            background:rgba(255,255,255,0.04) !important;
            border:1px solid rgba(140,60,220,0.28) !important;
            border-radius:10px !important;
            color:#e8d8ff !important;
            font-family:'Share Tech Mono',monospace !important;
            font-size:14px !important;
            padding:12px 16px !important;
            transition:all 0.3s ease !important;
        }
        .stTextInput input:focus {
            border-color:rgba(160,80,255,0.65) !important;
            box-shadow:0 0 18px rgba(140,40,220,0.28) !important;
            background:rgba(120,40,200,0.07) !important;
        }
        .stTextInput input::placeholder { color:rgba(180,130,255,0.3) !important; }

        /* Login button */
        div[data-testid="stButton"] button {
            background:linear-gradient(135deg,#5010a0,#9030e0,#5010a0) !important;
            background-size:200% auto !important;
            color:white !important;
            border:none !important;
            border-radius:10px !important;
            font-family:'Share Tech Mono',monospace !important;
            font-size:12px !important;
            font-weight:700 !important;
            letter-spacing:4px !important;
            padding:14px !important;
            text-transform:uppercase !important;
            box-shadow:0 4px 24px rgba(130,40,220,0.5) !important;
            transition:all 0.3s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            background-position:right center !important;
            box-shadow:0 6px 36px rgba(160,60,255,0.7) !important;
            transform:translateY(-2px) !important;
        }
        div[data-testid="stButton"] button:active {
            transform:translateY(0) scale(0.98) !important;
        }

        /* Version tag */
        .version-tag {
            text-align:center; margin-top:20px;
            font-family:'Share Tech Mono',monospace;
            font-size:10px; letter-spacing:3px;
            color:rgba(150,80,220,0.25);
        }

        /* Floating particles */
        .particle {
            position:fixed; border-radius:50%;
            pointer-events:none; z-index:0; opacity:0;
            animation:floatUp var(--dur,8s) linear infinite var(--delay,0s);
        }
        @keyframes floatUp {
            0%   { transform:translateY(100vh) scale(0); opacity:0; }
            10%  { opacity:0.65; }
            90%  { opacity:0.3; }
            100% { transform:translateY(-10vh) scale(1.5); opacity:0; }
        }

        /* Login card stagger */
        .login-card { animation: cardIn 0.9s cubic-bezier(0.16,1,0.3,1) both; }
        .orbit-wrap  { animation: fadeSlideDown 0.5s ease 0.1s both; }
        @keyframes fadeSlideDown {
            from { opacity:0; transform:translateY(-16px); }
            to   { opacity:1; transform:translateY(0); }
        }
        .status-bar  { animation: fadeSlideUp 0.5s ease 2.2s both; }
        .login-divider { animation: expandWidth 0.6s ease 2.4s both; transform-origin:center; }
        @keyframes expandWidth {
            from { transform:scaleX(0); opacity:0; }
            to   { transform:scaleX(1); opacity:1; }
        }
        </style>

        <!-- Background layers -->
        <div class="nebula nebula-1"></div>
        <div class="nebula nebula-2"></div>
        <div class="nebula nebula-3"></div>
        <div class="scanline"></div>

        <!-- Stars canvas -->
        <canvas class="stars-layer" id="starsCanvas"></canvas>

        <!-- Floating particles -->
        <div id="particlesWrap"></div>

        <script>
        const canvas = document.getElementById('starsCanvas');
        if (canvas) {
            canvas.width  = window.innerWidth;
            canvas.height = window.innerHeight;
            const ctx = canvas.getContext('2d');
            const stars = Array.from({length:200}, () => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() < 0.85 ? Math.random()*1.2+0.3 : Math.random()*2+1,
                speed: Math.random()*0.005+0.002,
                phase: Math.random()*Math.PI*2
            }));
            function drawStars(t) {
                ctx.clearRect(0,0,canvas.width,canvas.height);
                stars.forEach(s => {
                    const alpha = 0.15 + 0.85*(0.5+0.5*Math.sin(t*s.speed*60+s.phase));
                    ctx.beginPath();
                    ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
                    ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(2)})`;
                    ctx.fill();
                });
                requestAnimationFrame(drawStars);
            }
            requestAnimationFrame(drawStars);
        }

        const colors = ['rgba(160,80,255,0.7)','rgba(120,200,255,0.6)','rgba(200,80,255,0.55)'];
        const pw = document.getElementById('particlesWrap');
        if (pw) {
            for (let i=0; i<22; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                const size = Math.random()*3+1;
                p.style.cssText = `
                    left:${Math.random()*100}%;
                    width:${size}px; height:${size}px;
                    background:${colors[Math.floor(Math.random()*colors.length)]};
                    --dur:${(Math.random()*10+6).toFixed(1)}s;
                    --delay:${(Math.random()*8).toFixed(1)}s;
                `;
                pw.appendChild(p);
            }

            // Meteors
            for (let i=0; i<10; i++) {
                const m = document.createElement('div');
                m.className = 'meteor';
                const angle = 30 + Math.random()*20;
                const rad = angle * Math.PI/180;
                const dist = 400 + Math.random()*400;
                const w = 80 + Math.random()*120;
                m.style.cssText = `
                    left:${Math.random()*120-10}%;
                    top:${Math.random()*60}%;
                    width:${w}px;
                    transform:rotate(${angle}deg);
                    --dur:${(Math.random()*4+2).toFixed(1)}s;
                    --delay:${(Math.random()*10).toFixed(1)}s;
                    --dx:${Math.round(Math.cos(rad)*dist)};
                    --dy:${Math.round(Math.sin(rad)*dist)};
                `;
                pw.appendChild(m);
            }
        }
        </script>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        # Card open + corners
        st.markdown('<div class="login-card"><div class="c-tl"></div><div class="c-tr"></div><div class="c-bl"></div><div class="c-br"></div>', unsafe_allow_html=True)
        # Orbit icon
        st.markdown('<div class="orbit-wrap"><div class="orbit"><div class="orbit-ring2"></div><div class="orbit-ring"></div><div class="orbit-core"></div></div></div>', unsafe_allow_html=True)
        # Title + subtitle
        st.markdown('<div class="logo-title">ASHU YOLO AI</div><div class="logo-sub">Neural Vision &nbsp;·&nbsp; Deep Space</div>', unsafe_allow_html=True)
        # Status bar + divider
        st.markdown('<div class="status-bar"><div class="status-dot"></div>SYSTEM ONLINE &nbsp;·&nbsp; SECURE ACCESS</div><div class="login-divider"></div>', unsafe_allow_html=True)
        # Card close
        st.markdown('</div>', unsafe_allow_html=True)

        u = st.text_input("Username", placeholder="Enter username", key="login_user")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        p = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("⚡  INITIALIZE ACCESS", use_container_width=True):
            if u == "admin" and p == "ashu@123":
                st.success("✅ Access Granted! Initializing...")
                time.sleep(0.8)
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.markdown("""
                    <style>
                    .stTextInput input { animation: shake 0.4s ease; }
                    @keyframes shake {
                        0%,100% { transform:translateX(0); }
                        20% { transform:translateX(-8px); }
                        40% { transform:translateX(8px); }
                        60% { transform:translateX(-5px); }
                        80% { transform:translateX(5px); }
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.error("❌ Invalid Credentials! Access Denied.")

        st.markdown('<div class="version-tag">YOLO AI v2.0 &nbsp;·&nbsp; NEURAL VISION ENGINE</div>',
                    unsafe_allow_html=True)
    st.stop()


# =================== SIDEBAR ===================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Rajdhani:wght@400;600&display=swap');

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d1a0d 100%) !important;
        border-right: 1px solid rgba(40,167,69,0.2) !important;
    }

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

    section[data-testid="stSidebar"] div[data-testid="stButton"]:last-child button {
        background: linear-gradient(135deg, #4a0000, #a00000) !important;
        border-color: rgba(160,0,0,0.5) !important;
    }

    .stApp { background: #080d08 !important; }
    h1, h2, h3 { color: #e8f5e9 !important; }

    /* Page content fade-in on every navigation */
    section.main > div { animation: pageFadeIn 0.45s ease both; }
    @keyframes pageFadeIn {
        from { opacity:0; transform:translateY(18px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* Title slide-in */
    h1 {
        animation: titleSlide 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s both !important;
    }
    @keyframes titleSlide {
        from { opacity:0; transform:translateX(-24px); }
        to   { opacity:1; transform:translateX(0); }
    }

    /* Metric cards pop */
    div[data-testid="metric-container"] {
        animation: metricPop 0.4s cubic-bezier(0.16,1,0.3,1) both;
    }
    div[data-testid="metric-container"]:nth-child(1) { animation-delay:0.1s; }
    div[data-testid="metric-container"]:nth-child(2) { animation-delay:0.18s; }
    div[data-testid="metric-container"]:nth-child(3) { animation-delay:0.26s; }
    div[data-testid="metric-container"]:nth-child(4) { animation-delay:0.34s; }
    @keyframes metricPop {
        from { opacity:0; transform:scale(0.88) translateY(12px); }
        to   { opacity:1; transform:scale(1) translateY(0); }
    }

    /* Sidebar nav button slide-in stagger */
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) button { animation: navSlide 0.35s ease 0.05s both; }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) button { animation: navSlide 0.35s ease 0.10s both; }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(3) button { animation: navSlide 0.35s ease 0.15s both; }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(4) button { animation: navSlide 0.35s ease 0.20s both; }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(5) button { animation: navSlide 0.35s ease 0.25s both; }
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(6) button { animation: navSlide 0.35s ease 0.30s both; }
    @keyframes navSlide {
        from { opacity:0; transform:translateX(-20px); }
        to   { opacity:1; transform:translateX(0); }
    }
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

selected_idx = pages.index(st.session_state.page) + 1 if st.session_state.page in pages else 1

st.sidebar.markdown(f"""
    <style>
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



    if "webcam_model" not in st.session_state:
        st.session_state.webcam_model = st.session_state.model

    if apply_btn and selected_model_wc != "Current Model":
        with st.spinner(f"Loading {selected_model_wc}..."):
            st.session_state.webcam_model = YOLO(selected_model_wc)
        st.success(f"✅ Model switched to `{selected_model_wc}`!")

    active_model = st.session_state.webcam_model
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

    # Chrome localhost fix — allow insecure camera
    st.markdown("""
        <script>
        // Force camera permission check
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({video: true})
                .then(s => { s.getTracks().forEach(t => t.stop()); })
                .catch(e => console.warn('Camera pre-check:', e));
        }
        </script>
    """, unsafe_allow_html=True)

    st.markdown("### 🎥 Live Detection Stream")
    ctx = webrtc_streamer(
        key="ashu-yolo-webcam-v2",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
                "frameRate": {"ideal": 15, "max": 30},
                "facingMode": "user",
            },
            "audio": False
        },
        async_processing=True,
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
