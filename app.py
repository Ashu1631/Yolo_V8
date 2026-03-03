import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# DARK THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
h1,h2,h3,h4 {color: #00FFFF;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION INIT
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "model_object" not in st.session_state:
    st.session_state.model_object = None
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

# --------------------------------------------------
# PASSWORD HASH
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --------------------------------------------------
# LOAD USERS
# --------------------------------------------------
with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k.strip(): v['password'].strip()
             for k,v in users_yaml['users'].items()}

# --------------------------------------------------
# LOGIN
# --------------------------------------------------
def login():
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

if not st.session_state.logged_in:
    login()
    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
pages = ["Model Selection", "Upload & Detect",
         "Webcam Detection", "Evaluation Dashboard"]

st.session_state.page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state.page)
)

page = st.session_state.page

# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
if page == "Model Selection":
    st.title("📦 Model Selection")

    col1, col2 = st.columns(2)

    if os.path.exists("best.pt"):
        if col1.button("Load best.pt"):
            with st.spinner("Loading best.pt..."):
                st.session_state.model_object = load_model("best.pt")
            st.session_state.page = "Upload & Detect"
            st.rerun()

    if os.path.exists("yolov8.pt"):
        if col2.button("Load yolov8.pt"):
            with st.spinner("Loading yolov8.pt..."):
                st.session_state.model_object = load_model("yolov8.pt")
            st.session_state.page = "Upload & Detect"
            st.rerun()

# --------------------------------------------------
# UPLOAD & DETECT
# --------------------------------------------------
if page == "Upload & Detect":

    st.title("📤 Upload & Detect")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    uploaded_file = st.file_uploader(
        "Upload Image or Video", type=["jpg","png","jpeg","mp4"])

    if uploaded_file:

        temp_path = os.path.join(
            tempfile.gettempdir(), uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # ---------------- IMAGE ----------------
        if uploaded_file.name.endswith(("jpg","png","jpeg")):
            results = model(temp_path, conf=conf, iou=iou)
            img = results[0].plot()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)

        # ---------------- VIDEO (FAST) ----------------
        if uploaded_file.name.endswith("mp4"):
            cap = cv2.VideoCapture(temp_path)
            FRAME_WINDOW = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640,480))
                results = model(frame, conf=conf, iou=iou)
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated,
                                         cv2.COLOR_BGR2RGB)

                FRAME_WINDOW.image(annotated)

            cap.release()

# --------------------------------------------------
# WEBCAM DETECTION
# --------------------------------------------------
if page == "Webcam Detection":

    st.title("📷 Live Webcam Detection")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Webcam access failed.")
            st.stop()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640,480))
            results = model(frame)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated,
                                     cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(annotated)

        cap.release()

# --------------------------------------------------
# EVALUATION DASHBOARD
# --------------------------------------------------
if page == "Evaluation Dashboard":

    st.title("📊 Model Evaluation & Error Analysis")

    analysis_path = "analysis"

    metrics_file = os.path.join(analysis_path, "results.csv")

    if not os.path.exists(metrics_file):
        st.error("results.csv not found in analysis folder.")
        st.stop()

    df = pd.read_csv(metrics_file)

    precision_col = "metrics/precision(B)"
    recall_col = "metrics/recall(B)"
    map50_col = "metrics/mAP50(B)"
    map5095_col = "metrics/mAP50-95(B)"
    loss_col = "train/box_loss"

    latest = df.iloc[-1]

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("mAP50",
                f"{latest[map50_col]*100:.2f}%")
    col2.metric("mAP50-95",
                f"{latest[map5095_col]*100:.2f}%")
    col3.metric("Precision",
                f"{latest[precision_col]*100:.2f}%")
    col4.metric("Recall",
                f"{latest[recall_col]*100:.2f}%")

    st.subheader("Training Graphs")

    st.line_chart(df[[loss_col]].rename(
        columns={loss_col:"Box Loss"}))
    st.line_chart(df[[precision_col]].rename(
        columns={precision_col:"Precision"}))
    st.line_chart(df[[recall_col]].rename(
        columns={recall_col:"Recall"}))

    st.subheader("Error Analysis")

    if latest[precision_col] < 0.6:
        st.error("Low Precision → Too many False Positives")

    if latest[recall_col] < 0.6:
        st.error("Low Recall → Too many False Negatives")

    if latest[map50_col] < 0.5:
        st.warning("Low mAP → Improve dataset or train longer")

    for img in ["confusion_matrix.png",
                "PR_curve.png",
                "F1_curve.png",
                "results.png"]:
        path = os.path.join(analysis_path, img)
        if os.path.exists(path):
            st.image(path)
