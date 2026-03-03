import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard",
                   page_icon="🚀",
                   layout="wide")

# ==================================================
# OUTPUT FOLDER SETUP
# ==================================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# SESSION INIT
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "model_object" not in st.session_state:
    st.session_state.model_object = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

# ==================================================
# SIDEBAR STYLE
# ==================================================
st.sidebar.markdown("""
<style>
.nav-btn button {
    width: 100%;
    padding: 10px;
    margin-bottom: 6px;
    border-radius: 8px;
    cursor: pointer !important;
}
.nav-active button {
    width: 100%;
    padding: 10px;
    margin-bottom: 6px;
    border-radius: 8px;
    background-color: #00FFFF !important;
    color: black !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🚀 Navigation")

nav_options = ["Model Selection", "Upload & Detect",
               "Webcam Detection", "Evaluation Dashboard"]

for option in nav_options:
    container_class = "nav-active" if st.session_state.page == option else "nav-btn"
    with st.sidebar.container():
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        if st.button(option, key=f"nav_{option}", use_container_width=True):
            st.session_state.page = option
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state.page

# ==================================================
# LOGIN
# ==================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k.strip(): v['password'].strip()
             for k, v in users_yaml['users'].items()}

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users:
            stored = users[username]
            if stored == password or stored == hash_password(password):
                st.session_state.logged_in = True
                st.rerun()
        st.error("Invalid Credentials ❌")
    st.stop()

# ==================================================
# MODEL SELECTION
# ==================================================
if page == "Model Selection":
    st.title("📦 Model Selection")

    model_files = [f for f in os.listdir() if f.endswith(".pt")]

    selected_model = st.selectbox(
        "Please select the model",
        ["-- Select Model --"] + model_files
    )

    if selected_model != "-- Select Model --":
        with st.spinner(f"Loading {selected_model}..."):
            st.session_state.model_object = YOLO(selected_model)
            st.session_state.selected_model = selected_model
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def show_detection_counts(results):
    boxes = results[0].boxes
    counts = {}
    if boxes is not None and len(boxes.cls) > 0:
        classes = boxes.cls.cpu().numpy()
        names = results[0].names
        for c in classes:
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1
    return counts

def generate_detection_report(image_name, counts, detection_time):
    report_text = f"""
YOLOv8 Detection Report
----------------------------
Image Name: {image_name}
Detection Time: {detection_time:.2f} sec

Object Counts:
"""
    for obj, count in counts.items():
        report_text += f"{obj}: {count}\n"

    report_path = os.path.join(OUTPUT_DIR, f"report_{image_name}.txt")

    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path

# ==================================================
# UPLOAD & DETECT
# ==================================================
if page == "Upload & Detect":

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    uploaded_file = st.file_uploader(
        "Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:
        temp_path = os.path.join(
            tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # IMAGE DETECTION
        if uploaded_file.name.lower().endswith(("jpg", "png", "jpeg")):
            start = time.time()
            results = model(temp_path, conf=conf, iou=iou)
            end = time.time()

            img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            st.image(img)

            counts = show_detection_counts(results)
            st.subheader("📊 Detection Counts")
            st.json(counts)

            # Save Image
            output_path = os.path.join(
                OUTPUT_DIR, f"detected_{uploaded_file.name}")
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            with open(output_path, "rb") as file:
                st.download_button("⬇ Download Annotated Image",
                                   file,
                                   file_name=os.path.basename(output_path))

            # Generate Report
            report_path = generate_detection_report(
                uploaded_file.name, counts, end-start)

            with open(report_path, "rb") as file:
                st.download_button("📄 Download Detection Report",
                                   file,
                                   file_name=os.path.basename(report_path))

        # VIDEO DETECTION
        if uploaded_file.name.lower().endswith("mp4"):
            cap = cv2.VideoCapture(temp_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video_path = os.path.join(
                OUTPUT_DIR, f"detected_{uploaded_file.name}")

            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))
            FRAME_WINDOW = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, conf=conf, iou=iou)
                annotated = results[0].plot()
                out.write(annotated)
                FRAME_WINDOW.image(cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB))

            cap.release()
            out.release()

            with open(output_video_path, "rb") as file:
                st.download_button("⬇ Download Processed Video",
                                   file,
                                   file_name=os.path.basename(output_video_path))

    # DATASET DETECTION
    st.subheader("📂 Dataset Images")
    dataset_path = "datasets"

    if os.path.exists(dataset_path):
        dataset_images = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        selected_dataset_img = st.selectbox(
            "Select Dataset Image",
            ["-- Select Image --"] + dataset_images
        )

        if selected_dataset_img != "-- Select Image --":
            img_path = os.path.join(dataset_path, selected_dataset_img)
            results = model(img_path, conf=conf, iou=iou)
            img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            st.image(img)

# ==================================================
# WEBCAM (TURN RELAY + FPS)
# ==================================================
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model
        self.prev_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        annotated = results[0].plot()

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        cv2.putText(annotated, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

if page == "Webcam Detection":

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    webrtc_streamer(
        key="yolo-live",
        video_processor_factory=lambda: YOLOVideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceTransportPolicy": "relay",
            "iceServers": [{
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }]
        },
    )

# ==================================================
# EVALUATION DASHBOARD
# ==================================================
if page == "Evaluation Dashboard":

    analysis_path = "analysis"
    metrics_file = os.path.join(analysis_path, "results.csv")

    if not os.path.exists(metrics_file):
        st.error("results.csv not found.")
        st.stop()

    df = pd.read_csv(metrics_file)

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
    col2.metric("mAP50-95",
                f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
    col3.metric("Precision",
                f"{latest['metrics/precision(B)']*100:.2f}%")
    col4.metric("Recall",
                f"{latest['metrics/recall(B)']*100:.2f}%")

    st.line_chart(df[['train/box_loss']])
    st.line_chart(df[['metrics/precision(B)']])
    st.line_chart(df[['metrics/recall(B)']])

    for img in ["confusion_matrix.png",
                "PR_curve.png",
                "F1_curve.png",
                "results.png"]:
        path = os.path.join(analysis_path, img)
        if os.path.exists(path):
            st.image(path)
