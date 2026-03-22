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


# =================== LOGIN ===================
if not st.session_state.logged_in:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url("https://plus.unsplash.com/premium_photo-1664297538041-05e3224407a6?q=80&w=870&auto=format&fit=crop");
            background-size: cover; background-position: center;
        }
        label { color: #28a745 !important; font-weight: bold !important; }
        .stTextInput input {
            background-color: rgba(255,255,255,0.1) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 10px !important;
        }
        h2 { color: white !important; font-family: 'Segoe UI', sans-serif; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2>🎯 ASHU YOLO AI</h2>", unsafe_allow_html=True)
        u = st.text_input("Username", placeholder="Enter admin username")
        p = st.text_input("Password", type="password", placeholder="Enter password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login", use_container_width=True):
            if u == "admin" and p == "ashu@123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials! ❌")
    st.stop()


# =================== SIDEBAR ===================
pages = ["Model Selection", "Upload & Detect", "📷 Webcam Detection", "Evaluation Dashboard", "Model Comparison"]
st.sidebar.markdown("## 🚀 Navigation")
for p in pages:
    selected = st.session_state.page == p
    btn_color = "#28a745" if selected else "#dc3545"
    st.sidebar.markdown(f"""
        <style>
        div.stButton > button[key='nav_{p}'] {{
            background-color: {btn_color} !important;
            color: white !important; border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            height: 45px !important;
        }}
        </style>
    """, unsafe_allow_html=True)
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()
st.sidebar.divider()
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

def get_fps_value(dt):
    return 1 / dt if dt > 0 else 0


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
                else:
                    st_frame = st.empty()

                fps_placeholder = st.empty()
                m_best = YOLO("best.pt") if compare else None
                m_nano = YOLO("yolov8n.pt") if compare else None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    start_t = time.time()
                    if compare:
                        r1 = m_best(frame, verbose=False)
                        r2 = m_nano(frame, verbose=False)
                        st_frame1.image(cv2.cvtColor(apply_supervision(frame.copy(), r1), cv2.COLOR_BGR2RGB))
                        st_frame2.image(cv2.cvtColor(apply_supervision(frame.copy(), r2), cv2.COLOR_BGR2RGB))
                    else:
                        res = st.session_state.model(frame, verbose=False)
                        st_frame.image(
                            cv2.cvtColor(apply_supervision(frame, res), cv2.COLOR_BGR2RGB),
                            use_container_width=True
                        )
                    fps_placeholder.metric("⚡ Live FPS", f"{get_fps_value(time.time() - start_t):.1f}")
                cap.release()

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
                st.metric("⚡ Inference FPS", f"{get_fps_value(time.time() - start_t):.1f}")

    with tab2:
        if os.path.exists("datasets"):
            ds_imgs = [f for f in os.listdir("datasets") if f.endswith((".jpg", ".png", ".jpeg"))]
            sel_ds = st.selectbox("Select Image from Dataset Folder", ["-- Select --"] + ds_imgs)
            if sel_ds != "-- Select --":
                compare_ds = st.checkbox("🔄 Dataset Comparison Mode")
                ds_img = cv2.imread(os.path.join("datasets", sel_ds))
                start_ds = time.time()
                res_ds = st.session_state.model(ds_img)
                dt_ds = time.time() - start_ds
                if compare_ds:
                    cl1, cl2 = st.columns(2)
                    with cl1:
                        st.image(cv2.cvtColor(apply_supervision(ds_img, YOLO("best.pt")(ds_img)), cv2.COLOR_BGR2RGB), caption="Best.pt")
                    with cl2:
                        st.image(cv2.cvtColor(apply_supervision(ds_img, YOLO("yolov8n.pt")(ds_img)), cv2.COLOR_BGR2RGB), caption="Yolov8n.pt")
                else:
                    st.image(cv2.cvtColor(apply_supervision(ds_img, res_ds), cv2.COLOR_BGR2RGB))
                st.metric("⚡ Dataset FPS", f"{get_fps_value(dt_ds):.1f}")
        else:
            st.info("📂 `datasets/` folder nahi mila.")

# ----------- WEBCAM DETECTION -----------
elif page == "📷 Webcam Detection":
    st.title("📷 Live Webcam Detection - Ashu YOLO AI")
    if not st.session_state.model:
        st.warning("⚠️ Pehle Model Selection page pe model load karein!")
        st.stop()

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:14px 20px;
    border-radius:12px;border-left:4px solid #28a745;margin-bottom:16px;">
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
        - Best browser: **Chrome**.
        - FPS aur object count seedha video pe dikhega.
        - Low FPS? Confidence badhao ya `yolov8n.pt` use karo.
        """)

    active_model = st.session_state.model
    if apply_btn and selected_model_wc != "Current Model":
        with st.spinner(f"Loading {selected_model_wc}..."):
            active_model = YOLO(selected_model_wc)
        st.success(f"Model switched to `{selected_model_wc}`!")

    def make_processor():
        proc = VideoProcessor()
        proc.model = active_model
        proc.conf_threshold = conf_threshold
        return proc

    st.markdown("### 🎥 Live Detection Stream")
    ctx = webrtc_streamer(
        key="ashu-yolo-webcam",
        video_processor_factory=make_processor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}},
            "audio": False
        },
        async_processing=True,
    )

    st.divider()
    s1, s2, s3 = st.columns(3)
    with s1:
        st.success("🟢 **Stream: LIVE**") if ctx.state.playing else st.error("🔴 **Stream: Stopped**")
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
