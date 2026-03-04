import streamlit as st
import os
import cv2
import time
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ================= LOGIN =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", {})
else:
    users = {}

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in users and users[username]["password"] == password:

            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error("Invalid Credentials")

    st.stop()

# ================= SESSION =================
if "model" not in st.session_state:
    st.session_state.model = None

if "fps_history" not in st.session_state:
    st.session_state.fps_history = []

# ================= NAVIGATION =================
st.sidebar.title("🚀 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Model Selection",
        "Upload & Detect",
        "Evaluation Dashboard",
        "Failure Cases",
        "Model Comparison"
    ]
)

# ================= FAILURE EXTRACTION =================
def extract_failures(results, image, name):

    os.makedirs("failure_cases", exist_ok=True)

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:

        cv2.imwrite(f"failure_cases/{name}", image)

# ================= DETECTION SUMMARY =================
def detection_summary(results):

    counts = {}

    boxes = results[0].boxes

    if boxes is None:
        return counts

    classes = boxes.cls.cpu().numpy()
    names = results[0].names

    for c in classes:

        label = names[int(c)]

        counts[label] = counts.get(label, 0) + 1

    return counts

# ================= MODEL SELECTION =================
if page == "Model Selection":

    st.title("📦 Model Selection")

    models = [f for f in os.listdir() if f.endswith(".pt")]

    selected = st.selectbox("Select Model", ["-- Select --"] + models)

    if selected != "-- Select --":

        st.session_state.model = YOLO(selected)

        st.success("Model Loaded Successfully")

# ================= UPLOAD & DETECT =================
if page == "Upload & Detect":

    if not st.session_state.model:

        st.warning("Load model first")
        st.stop()

    model = st.session_state.model

    uploaded = st.file_uploader(
        "Upload Image / Video",
        type=["jpg","png","jpeg","mp4"]
    )

    if uploaded:

        os.makedirs("outputs", exist_ok=True)

        path = os.path.join("outputs", uploaded.name)

        with open(path, "wb") as f:
            f.write(uploaded.read())

        # IMAGE
        if path.endswith(("jpg","png","jpeg")):

            img = cv2.imread(path)

            start = time.time()

            r = model(img)

            detect_time = time.time() - start

            fps = 1/detect_time if detect_time>0 else 0

            st.session_state.fps_history.append(fps)

            annotated = r[0].plot()

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            extract_failures(r, img, uploaded.name)

            counts = detection_summary(r)

            if counts:

                df = pd.DataFrame({
                    "Class": list(counts.keys()),
                    "Count": list(counts.values())
                })

                st.subheader("Detection Report")

                st.dataframe(df)

        # VIDEO
        if path.endswith("mp4"):

            cap = cv2.VideoCapture(path)

            frame_box = st.empty()

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                r = model(frame)

                annotated = r[0].plot()

                frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            cap.release()

# ================= FPS GRAPH =================
if st.session_state.fps_history:

    fps_df = pd.DataFrame({
        "Frame": range(len(st.session_state.fps_history)),
        "FPS": st.session_state.fps_history
    })

    st.subheader("⚡ FPS Performance")

    st.plotly_chart(
        px.line(
            fps_df,
            x="Frame",
            y="FPS"
        ),
        use_container_width=True
    )

# ================= EVALUATION =================
if page == "Evaluation Dashboard":

    st.title("Evaluation Dashboard")

    if os.path.exists("analysis/results.csv"):

        df = pd.read_csv("analysis/results.csv")

        st.subheader("Train Loss")

        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])

        st.subheader("Validation Loss")

        st.line_chart(df[['val/box_loss','val/cls_loss','val/dfl_loss']])

# ================= FAILURE CASES =================
if page == "Failure Cases":

    st.title("Failure Cases")

    if os.path.exists("failure_cases"):

        files = os.listdir("failure_cases")

        if files:

            f = st.selectbox("Select Failure", files)

            st.image(os.path.join("failure_cases", f))

# ================= MODEL COMPARISON =================
if page == "Model Comparison":

    st.title("Model Comparison")

    comp_df = pd.DataFrame({

        "Metric": ["mAP50","mAP50-95","Recall"],

        "best.pt": [0.82,0.65,0.78],

        "yolov8n.pt": [0.74,0.55,0.70]

    })

    st.plotly_chart(px.bar(comp_df, x="Metric", y=["best.pt","yolov8n.pt"]))
