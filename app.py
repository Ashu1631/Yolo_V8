import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"
import streamlit as st
import pandas as pd
import cv2
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
from datetime import datetime
import supervision as sv
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Ashu YOLO Enterprise Pro", layout="wide", initial_sidebar_state="expanded")

# ================= 1. DIRECTORY SETUP =================
DIRS = ["outputs/images", "outputs/videos", "failure_cases", "analysis", "datasets"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Model Selection"
if "model" not in st.session_state: st.session_state.model = None
if "secondary_model" not in st.session_state: st.session_state.secondary_model = None
    
def get_sleek_plot(image, model):
    results = model(image, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(results)
    labels = [f"{model.names[class_id]} {confidence*100:.0f}%" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    
    # --- Dynamic Font Logic ---
    
    # Image ki width ke hisaab se font scale calculate karna (e.g., 1920px image pe bada font)
    img_width = image.shape[1]
    dynamic_scale = max(0.6, img_width / 1000) # Base scale 0.6, badi images pe automatically badhega
    dynamic_thickness = max(1, int(img_width / 500)) # Badi image pe zyada bold
    
    box_annotator = sv.BoxAnnotator(thickness=dynamic_thickness)
    
    label_annotator = sv.LabelAnnotator(
        text_scale=dynamic_scale,      # Ab ye fix nahi, dynamic hai!
        text_thickness=dynamic_thickness, 
        text_padding=int(10 * dynamic_scale), 
        text_color=sv.Color.WHITE, 
        border_radius=5
    )
    
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

# ================= 3. LOGIN SYSTEM (TECH BACKGROUND) =================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.8)), 
                        url('https://medium.com/datatobiz/top-6-leading-machine-learning-companies-in-india-in-2022-9c9a69c4ad26');
            background-size: cover;
            background-attachment: fixed;
        }
        .main-title {
            color: #00ffff; text-align: center; font-size: 3.5rem; font-weight: 800;
            text-shadow: 0 0 20px rgba(0,255,255,0.7); margin-top: 50px;
            margin-bottom: 20px;
        }
        /* 🟢 Dark Green Button Styling */
        div.stButton > button {
            background-color: #006400 !important; 
            color: white !important;
            border: 1px solid #00ff00 !important;
            font-weight: bold;
            height: 3em;
            margin-top: 20px;
        }
        /* Inputs ko thoda saaf dikhane ke liye shadow */
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid #00ffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🚀 Ashu YOLO Enterprise</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        # --- 1. Transparent Image (Directly on Background) ---
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_container_width=True)
        
        st.markdown("<h2 style='color:white; text-align:left- align;'>🔐 Secure Login</h2>", unsafe_allow_html=True)
        
        # --- 2. Input Fields with Placeholders ---
        user = st.text_input("Username", placeholder="Enter Username", label_visibility="collapsed")
        pw = st.text_input("Password", type="password", placeholder="Enter Password", label_visibility="collapsed")
        
        # --- 3. Dark Green Sign In Button ---
        if st.button("Sign In", use_container_width=True):
            if user == "admin" and pw == "ashu@1234":
                st.session_state.logged_in = True
                st.rerun()
            else: 
                st.error("Access Denied!")
                
    st.stop()

# ================= 4. NAVIGATION (Custom Styled & Icons) =================

nav_items = {
    "Model Selection": "📦",
    "Upload & Detect": "🔍",
    "Dataset Analysis": "📁",
    "Webcam Detection": "🎥",
    "Evaluation Dashboard": "📊",
    "Failure Cases": "❌",
    "Model Comparison": "⚖️"
}

st.markdown("""
    <style>
        div[role="radiogroup"] > label > div:first-child { 
            display: none !important; 
        }
        div[role="radiogroup"] > label {
            background-color: #800000 !important;
            border: 2px solid #000000 !important;
            border-radius: 12px 30px 30px 12px !important;
            padding: 10px 20px !important;
            margin-bottom: 10px !important;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        div[role="radiogroup"] > label p {
            color: #FFFFFF !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }
        div[role="radiogroup"] > label:has(input:checked) {
            background-color: #008000 !important;
            border: 2px solid #000000 !important;
            box-shadow: 0px 0px 20px rgba(0, 255, 255, 0.6) !important;
            transform: scale(1.05) translateX(5px) !important;
        }
        div[role="radiogroup"] > label:has(input:checked) p {
            color: #000000 !important;
        }
        [data-testid="stSidebar"] {
            min-width: 300px !important;
        }
    </style>
    """, unsafe_allow_html=True)

pages = list(nav_items.keys())
display_options = [f"{nav_items[p]} {p}" for p in pages]

if 'page' not in st.session_state: 
    st.session_state.page = "Model Selection"

selected_item = st.sidebar.radio(
    "🚀 Navigation", 
    display_options, 
    index=pages.index(st.session_state.page)
)

# FIXED: Defining current_page variable so below conditions work
current_page = selected_item.split(" ", 1)[-1]

if st.session_state.page != current_page:
    st.session_state.page = current_page
    st.rerun()

st.title(f"Current Page: {st.session_state.page}")

# ================= 5. PAGE CONTENT =================

if current_page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    col1, col2 = st.columns(2)
    with col1:
        primary = st.selectbox("Select Primary Model", ["-- Select --"] + models)
    with col2:
        secondary = st.selectbox("Select Secondary Model (For Comparison)", ["None"] + models)
    
    if st.button("Initialize Models"):
        if primary != "-- Select --":
            st.session_state.model = YOLO(primary)
            st.session_state.model_name = primary
            if secondary != "None":
                st.session_state.secondary_model = YOLO(secondary)
                st.session_state.secondary_name = secondary
            st.success(f"Loaded: {primary}")
            st.session_state.page = "Upload & Detect"
            st.rerun()

elif current_page == "Upload & Detect":
    st.title("🔍 Sleek Detection & Comparison Hub")
    if not st.session_state.model:
        st.warning("⚠️ Load a model first!")
        st.stop()
    
    uploaded = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
    
    if uploaded:
        temp_path = os.path.join("outputs", uploaded.name)
        with open(temp_path, "wb") as f: f.write(uploaded.getbuffer())
        is_video = uploaded.name.endswith(".mp4")

        # --- 1. PHOTO COMPARE (Side by Side with Model Labels) ---
        c1, c2 = st.columns(2)
        
        # Model A Header
        c1.markdown(f"<h3 style='text-align: center; color: #00ffff;'>🚀 Model A: {st.session_state.model_name}</h3>", unsafe_allow_html=True)
        # Model B Header
        c2.markdown(f"<h3 style='text-align: center; color: #ff4b4b;'>🔥 Model B: {st.session_state.get('secondary_name', 'Secondary')}</h3>", unsafe_allow_html=True)
        
        out1, out2 = c1.empty(), c2.empty()
        
        # --- 2. SCORE / CALCULATOR VALUES ---
        st.divider()
        st.subheader("⚡ Inference Score (Calculator)")
        m1, m2 = st.columns(2)
        val_a = m1.empty()
        val_b = m2.empty()

        # --- 3. GRAPH (Separate Lines for Models) ---
        st.divider()
        st.subheader("📊 Performance Timeline (FPS Graph)")
        fps_chart = st.empty()
        
        history_a = []
        history_b = []

        if is_video:
            cap1 = cv2.VideoCapture(temp_path)
            cap2 = cv2.VideoCapture(temp_path)
            
            while cap1.isOpened():
                t_start_a = time.time()
                r1, f1 = cap1.read()
                if not r1: break
                res1 = get_sleek_plot(f1, st.session_state.model)
                fps_a = 1.0 / (time.time() - t_start_a)
                
                t_start_b = time.time()
                r2, f2 = cap2.read()
                if not r2: break
                res2 = get_sleek_plot(f2, st.session_state.secondary_model if st.session_state.secondary_model else st.session_state.model)
                fps_b = 1.0 / (time.time() - t_start_b)

                # Update Images
                out1.image(res1, channels="BGR")
                out2.image(res2, channels="BGR")

                # Update Calculator Values
                val_a.metric(f"🚀 {st.session_state.model_name}", f"{fps_a:.2f} FPS")
                val_b.metric(f"🔥 {st.session_state.get('secondary_name', 'Model B')}", f"{fps_b:.2f} FPS")

                # Update Graph
                history_a.append(fps_a)
                history_b.append(fps_b)
                df_graph = pd.DataFrame({
                    st.session_state.model_name: history_a[-50:],
                    st.session_state.get('secondary_name', 'Model B'): history_b[-50:]
                })
                fps_chart.line_chart(df_graph)

            cap1.release(); cap2.release()
        else:
            # Static Image Logic
            img = cv2.imread(temp_path)
            t1 = time.time()
            res1 = get_sleek_plot(img, st.session_state.model)
            fps_a = 1.0 / (time.time() - t1)
            
            t2 = time.time()
            res2 = get_sleek_plot(img, st.session_state.secondary_model)
            fps_b = 1.0 / (time.time() - t2)

            out1.image(res1, channels="BGR")
            out2.image(res2, channels="BGR")
            val_a.metric(f"🚀 {st.session_state.model_name}", f"{fps_a:.2f} FPS")
            val_b.metric(f"🔥 {st.session_state.secondary_name}", f"{fps_b:.2f} FPS")
            fps_chart.line_chart(pd.DataFrame({"FPS": [fps_a, fps_b]}, index=[st.session_state.model_name, st.session_state.secondary_name]))

elif current_page == "Dataset Analysis":
    st.title("📁 Sleek Dataset Explorer")
    files = [f for f in os.listdir("datasets") if f.endswith(('.jpg', '.png'))]
    if files:
        sel_img = st.selectbox("Select Dataset Image", files)
        img = cv2.imread(os.path.join("datasets", sel_img))
        
        # 1. PHOTO COMPARE with Visual Titles
        c1, c2 = st.columns(2)
        c1.markdown(f"<h3 style='text-align: center; color: #00ffff;'>🚀 Model A: {st.session_state.model_name}</h3>", unsafe_allow_html=True)
        c2.markdown(f"<h3 style='text-align: center; color: #ff4b4b;'>🔥 Model B: {st.session_state.secondary_name}</h3>", unsafe_allow_html=True)
        
        t1 = time.time()
        res_a = get_sleek_plot(img, st.session_state.model)
        fps_a = 1.0 / (time.time() - t1)
        
        t2 = time.time()
        res_b = get_sleek_plot(img, st.session_state.secondary_model)
        fps_b = 1.0 / (time.time() - t2)

        c1.image(res_a, channels="BGR")
        c2.image(res_b, channels="BGR")

        # 2. SCORE CALCULATOR
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric(f"🚀 Score {st.session_state.model_name}", f"{fps_a:.2f} FPS")
        m2.metric(f"🔥 Score {st.session_state.secondary_name}", f"{fps_b:.2f} FPS")

        # 3. GRAPH
        st.divider()
        st.subheader("🚀 Speed Benchmark")
        df_bench = pd.DataFrame({
            "Model": [st.session_state.model_name, st.session_state.secondary_name],
            "FPS": [fps_a, fps_b]
        }).set_index("Model")
        st.line_chart(df_bench)
        
elif current_page == "Evaluation Dashboard":
    st.title("📊 4. Evaluation & Results")
    st.markdown("Is section mein model ki training performance aur metrics ka detailed analysis hai.")
    st.divider()
    st.header("4.1 Performance Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Loss Curve")
        if os.path.exists("analysis/loss_curve.png"):
            st.image("analysis/loss_curve.png", caption="Training & Validation Loss", use_container_width=True)
        else: st.info("analysis/loss_curve.png nahi mili.")
    with col2:
        st.markdown("### 🎯 Confusion Matrix")
        if os.path.exists("analysis/confusion_matrix.png"):
            st.image("analysis/confusion_matrix.png", caption="Prediction Errors per Class", use_container_width=True)
        else: st.info("analysis/confusion_matrix.png nahi mili.")
    st.divider()
    st.header("4.2 Accuracy Metrics")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 📈 Box F1 Curve")
        if os.path.exists("analysis/BoxF1_curve.png"):
            st.image("analysis/BoxF1_curve.png", caption="F1-Score vs Confidence", use_container_width=True)
        else: st.info("analysis/BoxF1_curve.png nahi mili.")
    with col4:
        st.markdown("### 🎯 Box PR Curve")
        if os.path.exists("analysis/BoxPR_curve.png"):
            st.image("analysis/BoxPR_curve.png", caption="Precision-Recall Tradeoff", use_container_width=True)
        else: st.info("analysis/BoxPR_curve.png nahi mili.")
    st.divider()
    st.header("4.3 Summary Table (Final Scores)")
    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        df.columns = df.columns.str.strip()
        last_results = df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP@.5", f"{last_results.get('metrics/mAP50(B)', 0):.3f}")
        m2.metric("mAP@.5:.95", f"{last_results.get('metrics/mAP50-95(B)', 0):.3f}")
        m3.metric("Precision", f"{last_results.get('metrics/precision(B)', 0):.3f}")
        m4.metric("Recall", f"{last_results.get('metrics/recall(B)', 0):.3f}")
        with st.expander("📂 Raw Training Logs Dekhen"):
            st.dataframe(df, use_container_width=True)
    else: st.error("analysis/results.csv file missing hai.")

elif current_page == "Webcam Detection":
    st.title(f"🎥 Live Feed: {st.session_state.get('model_name', 'Model')}")
    RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}]})
    class VideoProcessor(VideoProcessorBase):
        def __init__(self): self.model = st.session_state.get('model', None)
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.model is not None:
                results = self.model(img, conf=0.5) 
                annotated_frame = results[0].plot()
            else:
                annotated_frame = cv2.putText(img, "Model Not Loaded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    if 'model' in st.session_state and st.session_state.model is not None:
        webrtc_streamer(key="yolo-live-detection", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIG, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    else:
        st.error("❌ Model load nahi mila!")

elif current_page == "Model Comparison":
    st.title("⚖️ Advanced Benchmarking (10-Graph Matrix)")

    # --- 🤖 Machine Learning PNG/Box Section ---
    st.markdown("""
        <div style="background: rgba(0, 255, 255, 0.05); border: 2px solid #00ffff; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 30px;">
            <h3 style="color: #00ffff; margin-top: 15px;">Neural Network Analytics Hub</h3>
        </div>
    """, unsafe_allow_html=True)

    # DataFrame Logic
    m1 = st.session_state.model_name if st.session_state.get('model') else "yolov8n.pt"
    m2 = st.session_state.secondary_name if st.session_state.get('secondary_model') else "best.pt"

    df_bench = pd.DataFrame({
        "Model": [m1, m2, "Baseline"], 
        "Precision": [0.88, 0.72, 0.65], 
        "Recall": [0.84, 0.70, 0.60], 
        "mAP50": [0.91, 0.75, 0.68], 
        "Latency_ms": [15, 8, 5], 
        "F1": [0.86, 0.71, 0.62], 
        "Params_M": [8.5, 3.2, 1.0], 
        "mAP50-95": [0.65, 0.45, 0.35], 
        "Throughput": [65, 120, 200]
    })

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📊 1. Precision (**Bar Chart**)")
        st.plotly_chart(px.bar(df_bench, x="Model", y="Precision", color="Model", template="plotly_dark"), use_container_width=True)

        st.markdown("### 📈 2. Inference Flow (**Line Graph**)")
        st.plotly_chart(px.line(df_bench, x="Model", y="Latency_ms", markers=True), use_container_width=True)

        st.markdown("### 📉 3. mAP50 Performance (**Area Graph**)")
        st.plotly_chart(px.area(df_bench, x="Model", y="mAP50"), use_container_width=True)

        st.markdown("### 🧪 4. Detection Stages (**Funnel Chart**)")
        st.plotly_chart(px.funnel(dict(number=[100, 80, 60, 40], stage=["Input", "Boxes", "Conf", "Final"]), x='number', y='stage'), use_container_width=True)

        st.markdown("### 🌌 5. Weight vs Accuracy (**Scatter Plot**)")
        st.plotly_chart(px.scatter(df_bench, x="Params_M", y="mAP50", size="Latency_ms", color="Model"), use_container_width=True)

    with c2:
        st.markdown("### 🍕 6. Recall Distribution (**Pie/Donut Chart**)")
        st.plotly_chart(px.pie(df_bench, names="Model", values="Recall", hole=0.3), use_container_width=True)

        st.markdown("### 🧮 7. Metric Correlation (**Heatmap Graph**)")
        st.plotly_chart(px.imshow(df_bench.corr(numeric_only=True), text_auto=True), use_container_width=True)

        st.markdown("### 🌊 8. Model Gain (**Waterfall Chart**)")
        st.plotly_chart(go.Figure(go.Waterfall(x=df_bench["Model"], y=[0.68, 0.07, 0.16], measure=["relative"]*3)), use_container_width=True)

        st.markdown("### 🏆 9. F1 Harmonic Mean (**Color Bar Chart**)")
        st.plotly_chart(px.bar(df_bench, x="Model", y="F1", color="F1"), use_container_width=True)

        st.markdown("### 🏎️ 10. Throughput vs mAP (**Bubble Graph**)")
        st.plotly_chart(px.scatter(df_bench, x="Throughput", y="mAP50-95", size="Params_M", color="Model"), use_container_width=True)
