import streamlit as st
import os
import cv2
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ==========================================================
# SESSION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# NAVIGATION (Green Selected / Red Unselected / Compact Gap)
# ==========================================================
st.sidebar.markdown("## 🚀 Navigation")

pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases",
    "Model Comparison"
]

for p in pages:
    if st.session_state.page == p:
        st.sidebar.markdown(
            f"""
            <div style="background:#28a745;
                        padding:6px;
                        border-radius:6px;
                        color:white;
                        font-weight:600;
                        margin-bottom:3px;">
                👉 {p}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if st.sidebar.button(p, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

st.sidebar.markdown("""
<style>
div[data-testid="stButton"] button {
    background:#dc3545;
    color:white;
    padding:6px;
    border-radius:6px;
    margin-bottom:3px;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

# ==========================================================
# MODEL SELECTION
# ==========================================================
if page == "Model Selection":

    st.title("📦 Model Selection")

    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":
        st.session_state.model = YOLO(selected)
        st.success("Model Loaded Successfully ✅")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==========================================================
# AUTO FAILURE EXTRACTION
# ==========================================================
def extract_failures(results, image, conf_threshold=0.3):
    os.makedirs("failure_cases", exist_ok=True)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.imwrite("failure_cases/no_detection.jpg", image)
        return

    conf = boxes.conf.cpu().numpy()
    if any(conf < conf_threshold):
        cv2.imwrite("failure_cases/low_confidence.jpg", image)

# ==========================================================
# UPLOAD & DATASET
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    # ===================== UPLOAD =====================
    with tab1:

        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg","png","jpeg","mp4"])

        if uploaded:

            compare_upload = st.checkbox(
                "Enable Comparison (best.pt vs yolov8n.pt)",
                key="compare_upload"
            )

            temp_path = uploaded.name
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            if temp_path.endswith(("jpg","png","jpeg")):

                img = cv2.imread(temp_path)

                if compare_upload:
                    col1, col2 = st.columns(2)

                    col1.markdown("### 🟢 best.pt")
                    r1 = YOLO("best.pt")(img)
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown("### 🔵 yolov8n.pt")
                    r2 = YOLO("yolov8n.pt")(img)
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    results = model(img)
                    extract_failures(results, img)
                    st.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))

            if temp_path.endswith("mp4"):

                cap = cv2.VideoCapture(temp_path)
                os.makedirs("outputs", exist_ok=True)

                width = int(cap.get(3))
                height = int(cap.get(4))
                fps_original = cap.get(cv2.CAP_PROP_FPS)

                out = cv2.VideoWriter(
                    "outputs/processed_video.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_original,
                    (width,height)
                )

                frame_window = st.empty()
                fps_list = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    start = time.time()

                    if compare_upload:
                        m1 = YOLO("best.pt")
                        m2 = YOLO("yolov8n.pt")
                        r1 = m1(frame)
                        r2 = m2(frame)
                        annotated = np.hstack((r1[0].plot(), r2[0].plot()))
                    else:
                        r = model(frame)
                        annotated = r[0].plot()

                    fps = 1/(time.time()-start)
                    fps_list.append(fps)

                    cv2.putText(annotated,
                                f"FPS:{fps:.2f}",
                                (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(0,255,0),2)

                    out.write(annotated)
                    frame_window.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))

                cap.release()
                out.release()

                st.success("Video Saved Automatically ✅")

                with open("outputs/processed_video.mp4","rb") as file:
                    st.download_button("Download Video",
                                       data=file,
                                       file_name="processed_video.mp4")

                st.subheader("📈 FPS Graph")
                st.line_chart(pd.DataFrame({"FPS":fps_list}))

    # ===================== DATASET =====================
    with tab2:

        dataset_path = "datasets"
        if os.path.exists(dataset_path):

            images = [f for f in os.listdir(dataset_path)
                      if f.endswith(("jpg","png","jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":

                compare_dataset = st.checkbox(
                    "Enable Comparison",
                    key="compare_dataset"
                )

                img_path = os.path.join(dataset_path, selected_img)
                img = cv2.imread(img_path)

                if compare_dataset:
                    col1,col2 = st.columns(2)

                    col1.markdown("### 🟢 best.pt")
                    r1 = YOLO("best.pt")(img)
                    col1.image(cv2.cvtColor(r1[0].plot(),cv2.COLOR_BGR2RGB))

                    col2.markdown("### 🔵 yolov8n.pt")
                    r2 = YOLO("yolov8n.pt")(img)
                    col2.image(cv2.cvtColor(r2[0].plot(),cv2.COLOR_BGR2RGB))
                else:
                    results = model(img)
                    st.image(cv2.cvtColor(results[0].plot(),cv2.COLOR_BGR2RGB))

# ==========================================================
# WEBCAM (Improved Stability)
# ==========================================================
if page == "Webcam Detection":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    )

    model = st.session_state.model

    class Processor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=Processor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video":True,"audio":False},
        async_processing=True
    )

# ==========================================================
# EVALUATION DASHBOARD
# ==========================================================
if page == "Evaluation Dashboard":

    st.title("📊 Evaluation Dashboard")

    csv_path="analysis/results.csv"

    if os.path.exists(csv_path):

        df=pd.read_csv(csv_path)
        latest=df.iloc[-1]

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("📊 mAP50",f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("📊 mAP50-95",f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("🎯 Precision",f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("🔁 Recall",f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("📉 Loss Curve")
        st.line_chart(df[['train/box_loss',
                          'train/cls_loss',
                          'train/dfl_loss']])

        st.subheader("📈 Recall Curve")
        st.line_chart(df[['metrics/recall(B)']])

        if os.path.exists("analysis/confusion_matrix.png"):
            st.subheader("🔥 Confusion Matrix")
            st.image("analysis/confusion_matrix.png")

# ==========================================================
# FAILURE CASES
# ==========================================================
if page=="Failure Cases":

    st.title("⚠ Failure Cases")

    if os.path.exists("failure_cases"):
        files=os.listdir("failure_cases")
        if files:
            selected=st.selectbox("Select Failure",files)
            st.image(os.path.join("failure_cases",selected))
        else:
            st.info("No failure cases found.")

# ==========================================================
# MODEL COMPARISON (All Graphs)
# ==========================================================
if page=="Model Comparison":

    st.title("🚀 Model Comparison")

    if os.path.exists("analysis/results.csv"):

        df=pd.read_csv("analysis/results.csv")
        latest=df.iloc[-1]

        metrics={
            "mAP50":latest['metrics/mAP50(B)'],
            "Precision":latest['metrics/precision(B)'],
            "Recall":latest['metrics/recall(B)']
        }

        labels=list(metrics.keys())
        values=list(metrics.values())

        st.subheader("📈 Line Chart")
        st.line_chart(pd.DataFrame(metrics,index=[0]))

        st.subheader("📊 Bar Chart")
        st.plotly_chart(px.bar(x=labels,y=values))

        st.subheader("🥧 Pie Chart")
        st.plotly_chart(px.pie(names=labels,values=values))

        st.subheader("📊 Histogram")
        st.plotly_chart(px.histogram(df,x='metrics/mAP50(B)'))

        st.subheader("💧 Waterfall Chart")
        fig = go.Figure(go.Waterfall(
            x=labels,
            y=values,
            measure=["relative","relative","relative"]
        ))
        st.plotly_chart(fig)

        st.subheader("🔻 Funnel Chart")
        st.plotly_chart(px.funnel(x=values,y=labels))
