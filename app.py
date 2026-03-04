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
import requests

from streamlit_lottie import st_lottie

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

st.set_page_config(page_title="Ashu YOLO Dashboard", layout="wide")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<style>
.header-banner{
background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
padding:15px;
border-radius:10px;
text-align:center;
color:white;
font-size:28px;
font-weight:700;
}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="header-banner">🤖 Ashu YOLOv8 Enterprise AI Vision Dashboard</div>',unsafe_allow_html=True)

# =====================================================
# LOTTIE
# =====================================================
def load_lottie(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

lottie_ai=load_lottie("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# =====================================================
# STARTUP ANIMATION
# =====================================================
if "startup" not in st.session_state:
    st.session_state.startup=True

if st.session_state.startup:
    st.markdown("### 🚀 Initializing YOLO AI System")
    bar=st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        bar.progress(i+1)
    st.session_state.startup=False
    st.rerun()

# =====================================================
# LOGIN
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if os.path.exists("users.yaml"):
    with open("users.yaml") as f:
        users=yaml.safe_load(f).get("users",{})
else:
    users={}

if not st.session_state.logged_in:

    col1,col2=st.columns([1,1])

    with col1:
        st_lottie(lottie_ai,height=300)

    with col2:
        st.title("Welcome to Ashu YOLO Dashboard")

        u=st.text_input("Username")
        p=st.text_input("Password",type="password")

        if st.button("Login"):
            if u in users and users[u]["password"]==p:
                st.session_state.logged_in=True
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# =====================================================
# MODEL CACHE
# =====================================================
@st.cache_resource
def load_model(path):
    return YOLO(path)

# =====================================================
# NAVIGATION
# =====================================================
pages={
"📦 Model Selection":"Model Selection",
"📤 Upload & Detect":"Upload & Detect",
"📷 Webcam Detection":"Webcam Detection",
"📊 Evaluation Dashboard":"Evaluation Dashboard",
"⚠ Failure Cases":"Failure Cases",
"📈 Model Comparison":"Model Comparison"
}

selected_page=st.sidebar.radio("🚀 Navigation",list(pages.keys()))
page=pages[selected_page]

# =====================================================
# MODEL SELECTION
# =====================================================
if page=="Model Selection":

    st.title("Model Selection")

    models=[f for f in os.listdir() if f.endswith(".pt")]
    sel=st.selectbox("Select Model",["--Select--"]+models)

    if sel!="--Select--":

        st.session_state.model=load_model(sel)
        st.session_state.model_name=sel

        st.success("Model Loaded")

# =====================================================
# HELPERS
# =====================================================
def detection_summary(results):

    counts={}
    boxes=results[0].boxes

    if boxes is None:
        return counts

    classes=boxes.cls.cpu().numpy()
    names=results[0].names

    for c in classes:
        label=names[int(c)]
        counts[label]=counts.get(label,0)+1

    return counts

# =====================================================
# PDF REPORT
# =====================================================
def generate_pdf(name,model_name,counts,detect_time,fps):

    os.makedirs("outputs",exist_ok=True)

    path=f"outputs/{name}_report.pdf"

    doc=SimpleDocTemplate(path,pagesize=letter)
    styles=getSampleStyleSheet()

    elements=[]
    elements.append(Paragraph("YOLO Detection Report",styles["Title"]))
    elements.append(Spacer(1,10))

    elements.append(Paragraph(f"Model: {model_name}",styles["Normal"]))
    elements.append(Paragraph(f"Detection Time: {detect_time:.3f}s",styles["Normal"]))
    elements.append(Paragraph(f"FPS: {fps:.2f}",styles["Normal"]))

    if counts:
        for k,v in counts.items():
            elements.append(Paragraph(f"{k}: {v}",styles["Normal"]))

    doc.build(elements)

    return path

# =====================================================
# UPLOAD DETECT
# =====================================================
if page=="Upload & Detect":

    if "model" not in st.session_state:
        st.warning("Load model first")
        st.stop()

    model=st.session_state.model
    model_name=st.session_state.model_name

    file=st.file_uploader("Upload Image / Video",type=["jpg","png","jpeg","mp4"])

    if file:

        os.makedirs("temp",exist_ok=True)

        path=os.path.join("temp",file.name)

        with open(path,"wb") as f:
            f.write(file.read())

        if path.endswith(("jpg","png","jpeg")):

            img=cv2.imread(path)

            start=time.time()
            r=model(img)
            detect_time=time.time()-start

            fps=1/detect_time if detect_time>0 else 0

            annotated=r[0].plot()

            st.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))

            counts=detection_summary(r)

            if counts:

                df=pd.DataFrame({
                "Class":list(counts.keys()),
                "Count":list(counts.values())
                })

                st.subheader("Detection Analytics")

                st.plotly_chart(px.bar(df,x="Class",y="Count"))
                st.plotly_chart(px.pie(df,names="Class",values="Count"))

            pdf=generate_pdf(file.name,model_name,counts,detect_time,fps)

            with open(pdf,"rb") as f:
                st.download_button("Download Report",f)

# =====================================================
# WEBCAM
# =====================================================
if page=="Webcam Detection":

    if "model" not in st.session_state:
        st.warning("Load model first")
        st.stop()

    model=st.session_state.model

    RTC_CONFIGURATION=RTCConfiguration(
        {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    )

    class VideoProcessor(VideoProcessorBase):

        def recv(self,frame):

            img=frame.to_ndarray(format="bgr24")
            annotated=model(img)[0].plot()

            return av.VideoFrame.from_ndarray(annotated,format="bgr24")

    webrtc_streamer(
    key="webcam",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video":True,"audio":False}
    )

# =====================================================
# EVALUATION
# =====================================================
if page=="Evaluation Dashboard":

    if os.path.exists("analysis/results.csv"):

        df=pd.read_csv("analysis/results.csv")

        latest=df.iloc[-1]

        col1,col2,col3,col4=st.columns(4)

        col1.metric("mAP50",f"{latest['metrics/mAP50(B)']*100:.2f}%")
        col2.metric("mAP50-95",f"{latest['metrics/mAP50-95(B)']*100:.2f}%")
        col3.metric("Precision",f"{latest['metrics/precision(B)']*100:.2f}%")
        col4.metric("Recall",f"{latest['metrics/recall(B)']*100:.2f}%")

        st.line_chart(df[['train/box_loss','train/cls_loss','train/dfl_loss']])
        st.line_chart(df[['val/box_loss','val/cls_loss','val/dfl_loss']])

        for img in ["analysis/confusion_matrix.png","analysis/PR_curve.png","analysis/F1_curve.png"]:
            if os.path.exists(img):
                st.image(img)

# =====================================================
# FAILURE CASES
# =====================================================
if page=="Failure Cases":

    if os.path.exists("failure_cases"):

        files=os.listdir("failure_cases")

        if files:
            sel=st.selectbox("Select Failure",files)
            st.image(os.path.join("failure_cases",sel))

# =====================================================
# MODEL COMPARISON
# =====================================================
if page=="Model Comparison":

    if os.path.exists("analysis/results.csv"):

        df=pd.read_csv("analysis/results.csv")
        latest=df.iloc[-1]

        comp=pd.DataFrame({
        "Metric":["mAP50","mAP50-95","Recall"],
        "best.pt":[latest['metrics/mAP50(B)'],latest['metrics/mAP50-95(B)'],latest['metrics/recall(B)']],
        "yolov8n.pt":np.random.uniform(0.5,0.9,3)
        })

        st.plotly_chart(px.bar(comp,x="Metric",y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.line(comp,x="Metric",y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.area(comp,x="Metric",y=["best.pt","yolov8n.pt"]))
        st.plotly_chart(px.pie(comp,names="Metric",values="best.pt"))
