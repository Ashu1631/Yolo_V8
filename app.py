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
    if not st.session_state.model: st.warning("⚠️ Model load karein!"); st.stop()
    
    tab1, tab2 = st.tabs(["📤 File Upload", "📂 Dataset Explorer"])
    
    with tab1:
        file = st.file_uploader("Upload Image/Video", type=["jpg","png","jpeg","mp4"])
        if file:
            compare = st.checkbox("🔄 Enable Comparison (best.pt vs yolov8n.pt)")
            
            # --- VIDEO HANDLING ---
            if file.name.lower().endswith(".mp4"):
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(file.read())
                cap = cv2.VideoCapture(tfile.name)
                
                st_frame = st.empty()
                st_fps = st.empty()
                
                m_best = YOLO("best.pt") if compare else None
                m_nano = YOLO("yolov8n.pt") if compare else None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    start_t = time.time()
                    if compare:
                        r1, r2 = m_best(frame, verbose=False), m_nano(frame, verbose=False)
                        combined = np.hstack((apply_supervision(frame, r1), apply_supervision(frame, r2)))
                        st_frame.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        res = st.session_state.model(frame, verbose=False)
                        st_frame.image(cv2.cvtColor(apply_supervision(frame, res), cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    dt = time.time() - start_t
                    st_fps.plotly_chart(get_fps_chart(dt), use_container_width=True)
                cap.release()

            # --- IMAGE HANDLING ---
            else:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                start_t = time.time()
                if compare:
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(apply_supervision(img, YOLO("best.pt")(img)), cv2.COLOR_BGR2RGB), caption="Best Model")
                    c2.image(cv2.cvtColor(apply_supervision(img, YOLO("yolov8n.pt")(img)), cv2.COLOR_BGR2RGB), caption="Nano Model")
                else:
                    res = st.session_state.model(img)
                    st.image(cv2.cvtColor(apply_supervision(img, res), cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.plotly_chart(get_fps_chart(time.time() - start_t))
            
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
    st.markdown("Performance Curves: Loss Curve, Confusion Matrix, Box F1, aur PR Curves ka visual analysis.")
    st.divider()

    results_path = "analysis/results.csv"
    
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        latest = df.iloc[-1]

        # --- Section 1: Key Metrics ---
        st.subheader("🎯 Key Performance Indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.1f}%")
        m2.metric("mAP50-95", f"{latest['metrics/mAP50-95(B)']*100:.1f}%")
        m3.metric("Precision", f"{latest['metrics/precision(B)']*100:.1f}%")
        m4.metric("Recall", f"{latest['metrics/recall(B)']*100:.1f}%")
        
        st.divider()

        # --- Section 2: Training Progress (Loss Curves) ---
        st.subheader("📉 Training vs Validation Loss")
        l1, l2 = st.columns(2)
        with l1:
            st.write("**Box Loss** (Localization)")
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
        with l2:
            st.write("**Class Loss** (Classification)")
            st.line_chart(df[['train/cls_loss', 'val/cls_loss']])

        st.divider()

        # --- Section 3: Professional Curves (Images) ---
        st.subheader("🖼️ Detailed Analysis Curves")
        
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "F1 & PR Curves", "Validation Batches"])

        with tab1:
            if os.path.exists("analysis/confusion_matrix.png"):
                st.image("analysis/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("Confusion Matrix image not found.")

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                if os.path.exists("analysis/BoxF1_curve.png"):
                    st.image("analysis/BoxF1_curve.png", caption="F1 Confidence Curve", use_container_width=True)
            with col_b:
                if os.path.exists("analysis/BoxPR_curve.png"):
                    st.image("analysis/BoxPR_curve.png", caption="Precision-Recall Curve", use_container_width=True)

    else:
        st.error("Results file (analysis/results.csv) missing in 'analysis/' folder. Please train the model first.")

elif page == "Model Comparison":
    st.title("🚀 Ashu YOLO AI - 10-Graph Benchmarking")
    st.markdown("⚖️ Model Benchmarking (10-Graph Matrix) Advanced Plotly visualizations for Latency, Accuracy, and Throughput.")
    st.divider()

    # Sample Data (Aap ise apne real results se replace kar sakte hain)
    metrics = ["mAP50", "mAP50-95", "Precision", "Recall", "Inference(ms)"]
    comp_df = pd.DataFrame({
        "Metric": metrics,
        "best.pt": [0.85, 0.65, 0.88, 0.82, 12.5],
        "yolov8n.pt": [0.78, 0.58, 0.80, 0.75, 8.2]
    })

    # Data transformation for some plots
    df_melted = comp_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

    # --- ROW 1: The Heavy Hitters ---
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        # 1. Bar Comparison (Accuracy Metrics)
        acc_df = df_melted[df_melted["Metric"] != "Inference(ms)"]
        st.plotly_chart(px.bar(acc_df, x="Metric", y="Score", color="Model", barmode="group", 
                               title="1. Accuracy Metrics Comparison", color_discrete_sequence=px.colors.qualitative.Pastel))

    with r1_col2:
        # 2. Latency vs Accuracy Scatter (Efficiency Plot)
        # Low Latency + High mAP = Best Model
        scatter_data = pd.DataFrame({
            "Model": ["best.pt", "yolov8n.pt"],
            "mAP50": [0.85, 0.78],
            "Latency (ms)": [12.5, 8.2]
        })
        st.plotly_chart(px.scatter(scatter_data, x="Latency (ms)", y="mAP50", text="Model", size=[20, 15],
                                   title="2. Latency vs Accuracy (Sweet Spot Analysis)"))

    st.divider()

    # --- ROW 2: Distribution & Flow ---
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        # 3. Wagon Wheel Radar Chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=comp_df["best.pt"][:4], theta=metrics[:4], fill='toself', name='best.pt'))
        fig_radar.add_trace(go.Scatterpolar(r=comp_df["yolov8n.pt"][:4], theta=metrics[:4], fill='toself', name='yolov8n.pt'))
        fig_radar.update_layout(title="3. Performance Radar (mAP/P/R)")
        st.plotly_chart(fig_radar, use_container_width=True)

    with r2_col2:
        # 4. Metric Trend Line
        st.plotly_chart(px.line(acc_df, x="Metric", y="Score", color="Model", markers=True, title="4. Performance Trend"))

    # --- ROW 3: Advanced Analytics (Heatmaps & Distribution) ---
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        # 5. Throughput Heatmap (FPS Analysis)
        # FPS = 1000 / Latency
        fps_data = [[1000/12.5, 1000/15], [1000/8.2, 1000/10]] # Sample hardware variations
        st.plotly_chart(px.imshow(fps_data, labels=dict(x="Hardware", y="Model", color="FPS"),
                                  x=['GPU (T4)', 'CPU'], y=['best.pt', 'yolov8n.pt'],
                                  title="5. Throughput Heatmap (FPS)", text_auto=True))

    with r3_col2:
        # 6. Pie Distribution (Metric Weightage)
        st.plotly_chart(px.pie(comp_df[comp_df["Metric"] != "Inference(ms)"], names="Metric", values="best.pt", 
                               title="6. mAP Weightage Distribution", hole=0.4))

    # --- ROW 4: Statistical Variations ---
    r4_col1, r4_col2, r4_col3, r4_col4 = st.columns(4)

    with r4_col1:
        # 7. Box Plot
        st.plotly_chart(px.box(df_melted, y="Score", color="Model", title="7. Score Spread"))
    
    with r4_col2:
        # 8. Histogram
        st.plotly_chart(px.histogram(df_melted, x="Score", nbins=5, title="8. Value Dist."))

    with r4_col3:
        # 9. Violin Plot
        st.plotly_chart(px.violin(df_melted, y="Score", box=True, title="9. Density"))

    with r4_col4:
        # 10. Strip Plot
        st.plotly_chart(px.strip(df_melted, x="Model", y="Score", color="Metric", title="10. Metric Points"))
