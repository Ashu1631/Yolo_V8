import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="YOLOv8 Enterprise Dashboard",
                   page_icon="🚀",
                   layout="wide")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# SESSION INIT
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"
if "model_object" not in st.session_state:
    st.session_state.model_object = None

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
# SIDEBAR NAVIGATION (FIXED ACTIVE STYLE)
# ==================================================
st.sidebar.markdown("""
<style>
.stButton > button {
    width: 100%;
    padding: 10px;
    margin-bottom: 6px;
    border-radius: 8px;
    cursor: pointer !important;
}
.stButton > button:hover {
    background-color: #1f2937 !important;
}
.active-page > button {
    background-color: #00FFFF !important;
    color: black !important;
    font-weight: bold !important;
    border: 2px solid white !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🚀 Navigation")

pages = [
    "Model Selection",
    "Upload & Detect",
    "Webcam Detection",
    "Evaluation Dashboard",
    "Failure Cases"
]

for p in pages:
    container = st.sidebar.container()
    if st.session_state.page == p:
        container.markdown('<div class="active-page">', unsafe_allow_html=True)
        if container.button(p, key=p):
            st.session_state.page = p
            st.rerun()
        container.markdown("</div>", unsafe_allow_html=True)
    else:
        if container.button(p, key=p):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_counts(results):
    boxes = results[0].boxes
    counts = {}
    if boxes is not None and len(boxes.cls) > 0:
        classes = boxes.cls.cpu().numpy()
        names = results[0].names
        for c in classes:
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1
    return counts

def show_bar_chart(counts):
    if counts:
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values())
        ax.set_title("Detected Objects Per Class")
        st.pyplot(fig)

def generate_report(name, counts, time_taken):
    report = f"YOLO Detection Report\n\nFile: {name}\nTime: {time_taken:.2f} sec\n\nCounts:\n"
    for k, v in counts.items():
        report += f"{k}: {v}\n"
    path = os.path.join(OUTPUT_DIR, f"report_{name}.txt")
    with open(path, "w") as f:
        f.write(report)
    return path

# ==================================================
# MODEL SELECTION
# ==================================================
if page == "Model Selection":
    st.title("📦 Model Selection")
    models = [f for f in os.listdir() if f.endswith(".pt")]
    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":
        st.session_state.model_object = YOLO(selected)
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==================================================
# UPLOAD & DETECT
# ==================================================
if page == "Upload & Detect":

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    uploaded = st.file_uploader("Upload Image/Video",
                                type=["jpg", "png", "jpeg", "mp4"])

    if uploaded:
        path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())

        if uploaded.name.endswith(("jpg", "png", "jpeg")):
            start = time.time()
            results = model(path)
            end = time.time()

            img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            st.image(img)

            counts = get_counts(results)
            st.json(counts)
            show_bar_chart(counts)

            save_path = os.path.join(OUTPUT_DIR, f"detected_{uploaded.name}")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            st.download_button("Download Image",
                               open(save_path, "rb"),
                               file_name=os.path.basename(save_path))

            report_path = generate_report(uploaded.name, counts, end-start)
            st.download_button("Download Report",
                               open(report_path, "rb"),
                               file_name=os.path.basename(report_path))

            # Compare Upload
            compare_upload = st.checkbox("Compare best.pt vs yolov8n.pt")
            if compare_upload:
                model_best = YOLO("best.pt")
                model_nano = YOLO("yolov8n.pt")

                r1 = model_best(path)
                r2 = model_nano(path)

                img1 = cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                col1.image(img1, caption="best.pt")
                col2.image(img2, caption="yolov8n.pt")

    # ================= DATASET =================
    st.subheader("📂 Dataset Images")

    dataset_path = "datasets"

    if os.path.exists(dataset_path):
        dataset_images = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if dataset_images:
            selected_img = st.selectbox(
                "Select Dataset Image",
                ["-- Select Image --"] + dataset_images
            )

            if selected_img != "-- Select Image --":
                img_path = os.path.join(dataset_path, selected_img)
                results = model(img_path)
                img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(img)

                compare_dataset = st.checkbox("Compare Models (Dataset)")
                if compare_dataset:
                    model_best = YOLO("best.pt")
                    model_nano = YOLO("yolov8n.pt")

                    r1 = model_best(img_path)
                    r2 = model_nano(img_path)

                    img1 = cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB)

                    col1, col2 = st.columns(2)
                    col1.image(img1, caption="best.pt")
                    col2.image(img2, caption="yolov8n.pt")

# ==================================================
# WEBCAM (SAFE VERSION)
# ==================================================
class SafeDualProcessor(VideoProcessorBase):
    def __init__(self):
        self.model1 = YOLO("best.pt")
        self.model2 = YOLO("yolov8n.pt")
        self.prev = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        try:
            r1 = self.model1(img)
            r2 = self.model2(img)

            f1 = r1[0].plot()
            f2 = r2[0].plot()

            combined = np.hstack((f1, f2))

            now = time.time()
            fps = 1 / (now - self.prev)
            self.prev = now

            cv2.putText(combined, f"FPS:{int(fps)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(combined, format="bgr24")
        except:
            return frame

if page == "Webcam Detection":
    st.title("📷 Multi Model Live Comparison")

    webrtc_streamer(
        key="dual",
        video_processor_factory=SafeDualProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

# ==================================================
# EVALUATION DASHBOARD (RESTORED)
# ==================================================
if page == "Evaluation Dashboard":
    path = "analysis/results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col3.metric("Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.line_chart(df[['train/box_loss']])
        st.line_chart(df[['metrics/precision(B)']])
        st.line_chart(df[['metrics/recall(B)']])

        cm_csv = "analysis/confusion_matrix.csv"
        if os.path.exists(cm_csv):
            cm = pd.read_csv(cm_csv, index_col=0)
            fig = px.imshow(cm, text_auto=True,
                            color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("results.csv not found.")

# ==================================================
# FAILURE CASES
# ==================================================
if page == "Failure Cases":
    st.title("🚨 Failure Case Analysis")

    base_path = os.path.join("analysis", "failure_cases")

    case_type = st.selectbox(
        "Select Failure Type",
        ["False Positives", "False Negatives", "Small Objects"]
    )

    folder_map = {
        "False Positives": "false_positives",
        "False Negatives": "false_negatives",
        "Small Objects": "small_objects"
    }

    selected_folder = os.path.join(base_path, folder_map[case_type])

    if os.path.exists(selected_folder):
        images = [f for f in os.listdir(selected_folder)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if images:
            cols = st.columns(3)
            for i, img_name in enumerate(images):
                img_path = os.path.join(selected_folder, img_name)
                cols[i % 3].image(img_path,
                                  caption=img_name,
                                  use_container_width=True)
        else:
            st.info("No images found in this category.")
    else:
        st.info("Failure cases folder not found.")
