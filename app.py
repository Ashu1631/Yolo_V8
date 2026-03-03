import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile
import cv2
import hashlib
import numpy as np
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

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
             for k, v in users_yaml['users'].items()}

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
# SIDEBAR
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

# ==================================================
# 1️⃣ MODEL SELECTION (Dynamic + Fast)
# ==================================================
if page == "Model Selection":

    st.title("📦 Model Selection")

    model_files = [f for f in os.listdir() if f.endswith(".pt")]

    if not model_files:
        st.error("No .pt model files found.")
        st.stop()

    selected_model = st.selectbox("Select Model", model_files)

    if st.button("Load Model"):
        with st.spinner("Loading model..."):
            model_path = os.path.join(os.getcwd(), selected_model)
            st.session_state.model_object = load_model(model_path)

        st.success(f"{selected_model} loaded successfully!")
        st.session_state.page = "Upload & Detect"
        st.rerun()

# ==================================================
# 2️⃣ UPLOAD & DETECT (Optimized)
# ==================================================
if page == "Upload & Detect":

    st.title("📤 Upload & Detect")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    # ---------------- Upload Section ----------------
    uploaded_file = st.file_uploader(
        "Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])

    if uploaded_file:

        temp_path = os.path.join(
            tempfile.gettempdir(), uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # -------- IMAGE --------
        if uploaded_file.name.lower().endswith(("jpg", "png", "jpeg")):

            start = time.time()

            results = model(temp_path, conf=conf, iou=iou)
            img = results[0].plot()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            end = time.time()

            st.image(img)
            st.success(f"Detection Time: {round(end-start,2)} sec")

        # -------- VIDEO (FAST STREAM) --------
        if uploaded_file.name.lower().endswith("mp4"):

            cap = cv2.VideoCapture(temp_path)
            FRAME_WINDOW = st.image([])

            frame_skip = 2   # skip every 2 frames for speed

            count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                if count % frame_skip != 0:
                    continue

                frame = cv2.resize(frame, (640, 480))

                results = model(frame, conf=conf, iou=iou)
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated,
                                         cv2.COLOR_BGR2RGB)

                FRAME_WINDOW.image(annotated)

            cap.release()

    # ---------------- Dataset Dropdown ----------------
    st.subheader("Select Image From Dataset Folder")

    dataset_path = "datasets"

    if os.path.exists(dataset_path):

        images = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if images:

            selected_img = st.selectbox("Choose Dataset Image", images)

            if st.button("Detect Selected Image"):

                img_path = os.path.join(dataset_path, selected_img)

                results = model(img_path, conf=conf, iou=iou)
                img = results[0].plot()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img)

        else:
            st.info("No images in dataset folder.")

# ==================================================
# 3️⃣ WEBCAM (Streamlit Cloud Compatible)
# ==================================================
if page == "Webcam Detection":

    st.title("📷 Webcam Detection")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    camera_image = st.camera_input("Take a picture")

    if camera_image:

        file_bytes = camera_image.getvalue()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame, conf=conf, iou=iou)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated,
                                 cv2.COLOR_BGR2RGB)

        st.image(annotated)

# ==================================================
# 4️⃣ EVALUATION DASHBOARD (UNCHANGED)
# ==================================================
if page == "Evaluation Dashboard":

    st.title("📊 Model Evaluation")

    analysis_path = "analysis"

    metrics_file = os.path.join(analysis_path, "results.csv")

    if not os.path.exists(metrics_file):
        st.error("results.csv not found.")
        st.stop()

    df = pd.read_csv(metrics_file)

    precision_col = "metrics/precision(B)"
    recall_col = "metrics/recall(B)"
    map50_col = "metrics/mAP50(B)"
    map5095_col = "metrics/mAP50-95(B)"
    loss_col = "train/box_loss"

    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("mAP50", f"{latest[map50_col]*100:.2f}%")
    col2.metric("mAP50-95", f"{latest[map5095_col]*100:.2f}%")
    col3.metric("Precision", f"{latest[precision_col]*100:.2f}%")
    col4.metric("Recall", f"{latest[recall_col]*100:.2f}%")

    st.line_chart(df[[loss_col]].rename(
        columns={loss_col: "Box Loss"}))
    st.line_chart(df[[precision_col]].rename(
        columns={precision_col: "Precision"}))
    st.line_chart(df[[recall_col]].rename(
        columns={recall_col: "Recall"}))

    for img in ["confusion_matrix.png",
                "PR_curve.png",
                "F1_curve.png",
                "results.png"]:
        path = os.path.join(analysis_path, img)
        if os.path.exists(path):
            st.image(path)
