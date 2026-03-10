import os
import streamlit as st
import pandas as pd
import cv2
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import av
from datetime import datetime
import supervision as sv

# ================= 1. PAGE & DIRECTORY SETUP =================
st.set_page_config(page_title="Ashu YOLO Enterprise Pro", layout="wide", initial_sidebar_state="expanded")

DIRS = ["outputs", "failure_cases", "analysis", "datasets"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Initialize Session States safely
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None
if "model_name" not in st.session_state: st.session_state.model_name = "None"
if "secondary_model" not in st.session_state: st.session_state.secondary_model = None
if "secondary_name" not in st.session_state: st.session_state.secondary_name = "None"

def get_sleek_plot(image, model):
    if model is None:
        return image
    
    results = model(image, conf=0.3, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.6,       
        text_thickness=1,     
        text_padding=8,
        text_position=sv.Position.TOP_LEFT 
    )
    box_annotator = sv.BoxAnnotator()
    
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    return annotated_frame

# ================= 2. LOGIN SYSTEM =================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                        url('https://images.unsplash.com/photo-1726403846137-3c7ae90afedc?q=80&w=1332&auto=format&fit=crop');
            background-size: cover;
        }
        .main-title { color: #00ffff; text-align: center; font-size: 3rem; font-weight: 800; margin-top: -50px; text-shadow: 0 0 15px rgba(0,255,255,0.5); }
        .stTextInput > div > div { border: 1px solid #00ffff !important; background-color: rgba(0,0,0,0.5) !important; border-radius: 8px !important; }
        div.stButton > button { margin-top: 10px; background-color: #006400 !important; color: white !important; border: none !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🚀 Ashu YOLO Enterprise</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown(f'<img src="https://static.vecteezy.com/system/resources/thumbnails/010/851/451/small/abstract-technological-background-with-various-technological-elements-structure-pattern-technology-backdrop-png.png" style="width:150px; display:block; margin:auto; filter:drop-shadow(0 0 10px #00ffff);">', unsafe_allow_html=True)
        st.markdown("<h3 style='color:white; text-align:center;'>🔐 Secure Login</h3>", unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="admin", label_visibility="collapsed")
        pw = st.text_input("Password", type="password", placeholder="password", label_visibility="collapsed")
        
        if st.button("Sign In", use_container_width=True):
            if user == "admin" and pw == "ashu@1234":
                st.session_state.logged_in = True
                st.rerun()
            else: 
                st.error("Access Denied!")
    st.stop()

# ================= 3. NAVIGATION =================
nav_items = {
    "Model Selection": "📦",
    "Upload & Detect": "🔍",
    "Dataset Analysis": "📁",
    "Evaluation Dashboard": "📊",
    "Model Comparison": "⚖️"
}

st.sidebar.markdown("### 🚀 Navigation")
selected_page = st.sidebar.radio("Go to", list(nav_items.keys()), format_func=lambda x: f"{nav_items[x]} {x}")

if st.session_state.page != selected_page:
    st.session_state.page = selected_page

# ================= 4. PAGES =================

if st.session_state.page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    col1, col2 = st.columns(2)
    with col1:
        primary = st.selectbox("Select Primary Model", ["-- Select --"] + models)
    with col2:
        secondary = st.selectbox("Select Secondary Model (Optional)", ["None"] + models)
    
    if st.button("Initialize Models"):
        if primary != "-- Select --":
            st.session_state.model = YOLO(primary)
            st.session_state.model_name = primary
            if secondary != "None":
                st.session_state.secondary_model = YOLO(secondary)
                st.session_state.secondary_name = secondary
            else:
                st.session_state.secondary_model = None
                st.session_state.secondary_name = "Not Selected"
            st.success(f"Loaded: {primary}")
            st.rerun()

elif st.session_state.page == "Upload & Detect":
    st.title("🔍 Detection & Comparison Hub")
    if not st.session_state.model:
        st.warning("⚠️ Please load a model in 'Model Selection' first!")
        st.stop()
    
    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    
    if uploaded:
        temp_path = os.path.join("outputs", uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())
        is_video = uploaded.name.endswith(".mp4")

        c1, c2 = st.columns(2)
        c1.markdown(f"<h4 style='text-align: center; color: #00ffff;'>Model A: {st.session_state.model_name}</h4>", unsafe_allow_html=True)
        c2.markdown(f"<h4 style='text-align: center; color: #ff4b4b;'>Model B: {st.session_state.secondary_name}</h4>", unsafe_allow_html=True)
        
        out1, out2 = c1.empty(), c2.empty()
        val_a, val_b = st.columns(2)[0].empty(), st.columns(2)[1].empty()
        fps_chart = st.empty()

        if is_video:
            cap = cv2.VideoCapture(temp_path)
            hist_a, hist_b = [], []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Model A Inference
                t_s1 = time.time()
                res1 = get_sleek_plot(frame, st.session_state.model)
                fps_a = 1.0 / (time.time() - t_s1 + 1e-6)
                
                # Model B Inference
                t_s2 = time.time()
                res2 = get_sleek_plot(frame, st.session_state.secondary_model) if st.session_state.secondary_model else frame
                fps_b = 1.0 / (time.time() - t_s2 + 1e-6)

                out1.image(res1, channels="BGR")
                out2.image(res2, channels="BGR")
                
                hist_a.append(fps_a)
                hist_b.append(fps_b)
                
                df_fps = pd.DataFrame({st.session_state.model_name: hist_a[-30:], 
                                      st.session_state.secondary_name: hist_b[-30:]})
                fps_chart.line_chart(df_fps)
            cap.release()
        else:
            img = cv2.imread(temp_path)
            res1 = get_sleek_plot(img, st.session_state.model)
            res2 = get_sleek_plot(img, st.session_state.secondary_model) if st.session_state.secondary_model else img
            out1.image(res1, channels="BGR")
            out2.image(res2, channels="BGR")

elif st.session_state.page == "Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard")
    st.markdown("---")
    
    # Check for results.csv
    csv_path = "analysis/results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        last = df.iloc[-1]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP@50", f"{last.get('metrics/mAP50(B)', 0):.3f}")
        m2.metric("mAP@50-95", f"{last.get('metrics/mAP50-95(B)', 0):.3f}")
        m3.metric("Precision", f"{last.get('metrics/precision(B)', 0):.3f}")
        m4.metric("Recall", f"{last.get('metrics/recall(B)', 0):.3f}")
        
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("💡 Place 'results.csv' in the 'analysis/' folder to see metrics.")

elif st.session_state.page == "Model Comparison":
    st.title("⚖️ Advanced Benchmarking")
    # Mock data for visualization
    m1 = st.session_state.model_name
    m2 = st.session_state.secondary_name
    
    df_bench = pd.DataFrame({
        "Model": [m1, m2],
        "Precision": [0.88, 0.75],
        "Recall": [0.82, 0.70],
        "mAP50": [0.90, 0.78],
        "Latency_ms": [12, 25]
    })
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df_bench, x="Model", y="mAP50", color="Model", title="Accuracy (mAP50)"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(df_bench, x="Model", y="Latency_ms", title="Latency (Lower is better)"), use_container_width=True)
