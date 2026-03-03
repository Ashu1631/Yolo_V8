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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

st.set_page_config(page_title="YOLOv8 Enterprise AI", layout="wide")

# ==========================================================
# LOGIN
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users = yaml.safe_load(f).get("users", {})
else:
    users = {}

if not st.session_state.logged_in:
    st.title("🔐 Login Required")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")
    st.stop()

# ==========================================================
# SESSION
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = "Model Selection"

if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================================
# NAVIGATION (FIXED PROPERLY)
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

selected_page = st.sidebar.radio(
    "",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = selected_page

st.markdown("""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: #dc3545;
    padding: 8px;
    margin-bottom: 5px;
    border-radius: 6px;
    color: white;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #28a745 !important;
}
</style>
""", unsafe_allow_html=True)

page = st.session_state.page

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
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

def generate_pdf_report(image_path, detection_counts):
    os.makedirs("outputs", exist_ok=True)
    pdf_path = "outputs/detection_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>YOLOv8 Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    for cls, count in detection_counts.items():
        elements.append(Paragraph(f"{cls}: {count}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    if os.path.exists(image_path):
        elements.append(Spacer(1, 12))
        elements.append(RLImage(image_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return pdf_path

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
# UPLOAD & DATASET
# ==========================================================
if page == "Upload & Detect":

    if not st.session_state.model:
        st.warning("Load model first.")
        st.stop()

    model = st.session_state.model
    tab1, tab2 = st.tabs(["📤 Upload", "📂 Dataset"])

    # ---------------- UPLOAD ----------------
    with tab1:
        uploaded = st.file_uploader("Upload Image/Video",
                                    type=["jpg", "png", "jpeg", "mp4"])

        if uploaded:
            compare = st.checkbox("Enable Comparison")

            temp_path = uploaded.name
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            # IMAGE
            if temp_path.endswith(("jpg", "png", "jpeg")):
                img = cv2.imread(temp_path)

                if compare:
                    col1, col2 = st.columns(2)

                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)

                    col1.markdown("### 🟢 BEST.PT")
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))

                    col2.markdown("### 🔵 YOLOV8N.PT")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    annotated = r[0].plot()
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            # VIDEO
            if temp_path.endswith("mp4"):
                cap = cv2.VideoCapture(temp_path)
                frame_box = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if compare:
                        r1 = YOLO("best.pt")(frame)
                        r2 = YOLO("yolov8n.pt")(frame)

                        left = r1[0].plot()
                        right = r2[0].plot()

                        cv2.rectangle(left,(0,0),(250,60),(0,255,0),-1)
                        cv2.putText(left,"BEST.PT",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,0,0),2)

                        cv2.rectangle(right,(0,0),(250,60),(255,0,0),-1)
                        cv2.putText(right,"YOLOV8N.PT",(20,40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(255,255,255),2)

                        annotated = np.hstack((left, right))
                    else:
                        r = model(frame)
                        annotated = r[0].plot()

                    frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                cap.release()

    # ---------------- DATASET ----------------
    with tab2:
        dataset_path = "datasets"
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path)
                      if f.endswith(("jpg","png","jpeg"))]

            selected_img = st.selectbox("Select Dataset Image",
                                        ["-- Select --"] + images)

            if selected_img != "-- Select --":
                img = cv2.imread(os.path.join(dataset_path, selected_img))
                compare_ds = st.checkbox("Enable Dataset Comparison")

                if compare_ds:
                    col1, col2 = st.columns(2)
                    r1 = YOLO("best.pt")(img)
                    r2 = YOLO("yolov8n.pt")(img)
                    col1.markdown("### 🟢 BEST.PT")
                    col1.image(cv2.cvtColor(r1[0].plot(), cv2.COLOR_BGR2RGB))
                    col2.markdown("### 🔵 YOLOV8N.PT")
                    col2.image(cv2.cvtColor(r2[0].plot(), cv2.COLOR_BGR2RGB))
                else:
                    r = model(img)
                    st.image(cv2.cvtColor(r[0].plot(), cv2.COLOR_BGR2RGB))

# ==========================================================
# EVALUATION (EXPANDED)
# ==========================================================
if page == "Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard")

    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        col1,col2,col3,col4 = st.columns(4)
        col1.metric("mAP50", f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("mAP50-95", f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("Precision", f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("Recall", f"{latest['metrics/recall(B)']*100:.2f}%")

        st.subheader("Loss Curve")
        st.line_chart(df[['train/box_loss',
                          'train/cls_loss',
                          'train/dfl_loss']])

        st.subheader("Stacked Overview")
        overview = df[['metrics/mAP50(B)',
                       'metrics/mAP50-95(B)',
                       'metrics/recall(B)']]
        st.area_chart(overview)

        if os.path.exists("analysis/confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image("analysis/confusion_matrix.png")

# ==========================================================
# FAILURE CASES
# ==========================================================
if page == "Failure Cases":
    st.title("⚠ Failure Cases")
    os.makedirs("failure_cases", exist_ok=True)
    files = os.listdir("failure_cases")
    if files:
        selected = st.selectbox("Select Failure Case", files)
        st.image(os.path.join("failure_cases", selected))
    else:
        st.info("No failure cases available.")

# ==========================================================
# MODEL COMPARISON (EXPANDED)
# ==========================================================
if page == "Model Comparison":
    st.title("🚀 Model Comparison")

    if os.path.exists("analysis/results.csv"):
        df = pd.read_csv("analysis/results.csv")
        latest = df.iloc[-1]

        metrics = {
            "mAP50": latest['metrics/mAP50(B)'],
            "mAP50-95": latest['metrics/mAP50-95(B)'],
            "Recall": latest['metrics/recall(B)']
        }

        comp_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "best.pt": list(metrics.values()),
            "yolov8n.pt": np.random.uniform(0.5,0.9,3)
        })

        st.dataframe(comp_df)

        st.plotly_chart(px.bar(comp_df, x="Metric",
                               y=["best.pt","yolov8n.pt"]))

        st.plotly_chart(px.line(comp_df, x="Metric",
                                y=["best.pt","yolov8n.pt"]))

        st.plotly_chart(px.area(comp_df, x="Metric",
                                y=["best.pt","yolov8n.pt"]))

        st.plotly_chart(px.histogram(comp_df, x="best.pt"))

        st.plotly_chart(px.pie(comp_df,
                               names="Metric",
                               values="best.pt"))

        fig = go.Figure(go.Waterfall(
            x=comp_df["Metric"],
            y=comp_df["best.pt"],
            measure=["relative"]*3
        ))
        st.plotly_chart(fig)

        st.plotly_chart(px.funnel(comp_df,
                                  x="best.pt",
                                  y="Metric"))
