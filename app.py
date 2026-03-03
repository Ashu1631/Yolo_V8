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
.stButton>button {background-color:#1f77b4; color:white;}
.stDownloadButton>button {background-color:#17becf; color:white;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
for key in ["logged_in", "selected_model", "model_object", "page"]:
    if key not in st.session_state:
        st.session_state[key] = False if key=="logged_in" else None

if st.session_state.page is None:
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
st.sidebar.title("🚀 YOLOv8 Enterprise Dashboard")

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

    model_files = [f for f in os.listdir() if f.endswith(".pt")]

    if not model_files:
        st.error("No .pt model files found!")
        st.stop()

    selected = st.selectbox("Select YOLO Model", model_files)

    if st.button("Load Model"):
        model_path = os.path.join(os.getcwd(), selected)
        st.session_state.selected_model = selected
        st.session_state.model_object = load_model(model_path)
        st.success(f"Model {selected} loaded successfully!")
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

    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45)

    option = st.radio("Input Type",
                      ["Upload Image/Video", "Use Dataset Folder"])

    if option == "Upload Image/Video":
        uploaded_file = st.file_uploader(
            "Upload File", type=["jpg","png","jpeg","mp4"])

        if uploaded_file:
            temp_path = os.path.join(
                tempfile.gettempdir(), uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            st.info("Running Detection...")

            results = model(temp_path, conf=conf,
                            iou=iou, save=True)

            save_dir = results[0].save_dir

            for file in os.listdir(save_dir):
                path = os.path.join(save_dir, file)
                if file.endswith((".jpg",".png")):
                    st.image(path)
                if file.endswith(".mp4"):
                    st.video(path)

    if option == "Use Dataset Folder":
        dataset_path = "datasets"

        if not os.path.exists(dataset_path):
            st.error("datasets folder not found!")
            st.stop()

        images = [os.path.join(dataset_path,f)
                  for f in os.listdir(dataset_path)
                  if f.lower().endswith((".jpg",".png",".jpeg"))]

        for img in images:
            results = model(img, conf=conf, iou=iou)
            for r in results:
                st.image(r.plot(),
                         caption=os.path.basename(img))

# --------------------------------------------------
# WEBCAM DETECTION
# --------------------------------------------------
if page == "Webcam Detection":
    st.title("📷 Live Webcam Detection")

    if not st.session_state.model_object:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model_object

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    iou = st.slider("IoU", 0.0, 1.0, 0.45)

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam access failed.")
                break

            results = model(frame, conf=conf, iou=iou)
            annotated = results[0].plot()

            FRAME_WINDOW.image(annotated)

        cap.release()

# --------------------------------------------------
# EVALUATION DASHBOARD
# --------------------------------------------------
if page == "Evaluation Dashboard":
    st.title("📊 Model Evaluation & Error Analysis")

    analysis_path = st.text_input(
        "Training Folder Path",
        "runs/detect/train"
    )

    if not os.path.exists(analysis_path):
        st.error("Invalid folder path.")
        st.stop()

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

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("mAP50",
                f"{latest[map50_col]*100:.2f}%")
    col2.metric("mAP50-95",
                f"{latest[map5095_col]*100:.2f}%")
    col3.metric("Precision",
                f"{latest[precision_col]*100:.2f}%")
    col4.metric("Recall",
                f"{latest[recall_col]*100:.2f}%")

    st.subheader("📈 Training Graphs")
    st.line_chart(df[[loss_col]].rename(
        columns={loss_col:"Box Loss"}))
    st.line_chart(df[[precision_col]].rename(
        columns={precision_col:"Precision"}))
    st.line_chart(df[[recall_col]].rename(
        columns={recall_col:"Recall"}))

    st.subheader("🔍 Error Analysis")

    if latest[precision_col] < 0.6:
        st.error("Low Precision → Too many False Positives")

    if latest[recall_col] < 0.6:
        st.error("Low Recall → Too many False Negatives")

    if latest[map50_col] < 0.5:
        st.warning("Low mAP → Improve dataset or train longer")

    # Show confusion & PR curves
    for img in ["confusion_matrix.png",
                "PR_curve.png",
                "F1_curve.png",
                "results.png"]:
        path = os.path.join(analysis_path, img)
        if os.path.exists(path):
            st.image(path)
