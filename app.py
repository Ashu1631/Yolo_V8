import streamlit as st
import os
import yaml
import pandas as pd
from ultralytics import YOLO
import tempfile

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# CUSTOM DARK STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: white;}
h1,h2,h3,h4 {color: #00FFFF;}
.stButton>button {background-color:#1f77b4; color:white;}
.stDownloadButton>button {background-color:#17becf; color:white;}
.metric {background-color: #1f1f1f; padding: 10px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION INIT
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# -----------------------------
# LOAD USERS (NESTED YAML SUPPORT)
# -----------------------------
with open("users.yaml") as file:
    users_yaml = yaml.safe_load(file)
    users = {k.strip(): v['password'].strip() for k, v in users_yaml['users'].items()}

# -----------------------------
# LOGIN FUNCTION
# -----------------------------
def login():
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username.strip() in users and users[username.strip()] == password.strip():
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.experimental_rerun()
        else:
            st.error("Invalid Credentials ❌")

# -----------------------------
# LOGIN CHECK
# -----------------------------
if not st.session_state.logged_in:
    login()
    st.stop()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🚀 YOLOv8 Enterprise Dashboard")
page = st.sidebar.radio("Navigation", ["Model Selection", "Upload & Detect", "Evaluation Dashboard"])

# -----------------------------
# MODEL SELECTION
# -----------------------------
if page == "Model Selection":
    st.title("📦 Model Selection")
    model_files = [f for f in os.listdir() if f.endswith(".pt")]
    if model_files:
        selected = st.selectbox("Select YOLO Model", model_files)
        st.session_state.selected_model = selected
        st.success(f"Selected Model: {selected}")
    else:
        st.error("No .pt model files found!")

# -----------------------------
# UPLOAD & DETECT
# -----------------------------
if page == "Upload & Detect":
    st.title("📤 Upload & Detect")
    if not st.session_state.selected_model:
        st.warning("⚠ Please select model first.")
        st.stop()

    option = st.radio("Choose Input Type", ["Upload Image/Video", "Use Dataset Folder"])
    model_path = os.path.join(os.getcwd(), st.session_state.selected_model)

    # SAFE MODEL LOADING
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model '{st.session_state.selected_model}': {e}")
        st.stop()

    # Upload File
    if option == "Upload Image/Video":
        uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.info("Running Detection...")
            results = model(tfile.name)
            for r in results:
                im_array = r.plot()
                st.image(im_array, caption="Detection Result", use_container_width=True)

    # Dataset Folder
    if option == "Use Dataset Folder":
        dataset_path = "datasets"
        if os.path.exists(dataset_path):
            st.success("Using images from datasets folder")
            images = [os.path.join(dataset_path, f)
                      for f in os.listdir(dataset_path)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            for img_path in images:
                results = model(img_path)
                for r in results:
                    im_array = r.plot()
                    st.image(im_array, caption=os.path.basename(img_path), use_container_width=True)
        else:
            st.error("Datasets folder not found!")

# -----------------------------
# EVALUATION DASHBOARD
# -----------------------------
if page == "Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard")
    analysis_path = "analysis"
    if not os.path.exists(analysis_path):
        st.error("Analysis folder not found!")
        st.stop()

    metrics_file = os.path.join(analysis_path, "results.csv")
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        latest = df.iloc[-1]  # last epoch metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP50", f"{latest['mAP50']*100:.2f}%")
        col2.metric("mAP50-95", f"{latest['mAP50-95']*100:.2f}%")
        col3.metric("Precision", f"{latest['precision']*100:.2f}%")
        col4.metric("Recall", f"{latest['recall']*100:.2f}%")

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        for fname in ["results.png", "PR_curve.png", "F1_curve.png"]:
            fpath = os.path.join(analysis_path, fname)
            if os.path.exists(fpath):
                st.image(fpath, caption=fname)
    with col2:
        fpath = os.path.join(analysis_path, "confusion_matrix.png")
        if os.path.exists(fpath):
            st.image(fpath, caption="Confusion Matrix")

    # Display separate metric charts
    if os.path.exists(metrics_file):
        st.subheader("📈 Training Metrics Over Epochs")
        st.line_chart(df[['loss']].rename(columns={'loss':'Loss'}))
        st.line_chart(df[['precision']].rename(columns={'precision':'Precision'}))
        st.line_chart(df[['recall']].rename(columns={'recall':'Recall'}))

        # Download CSV
        with open(metrics_file, "rb") as file:
            st.download_button("⬇ Download Results CSV", data=file, file_name="results.csv")

    # Download analysis images
    for fname in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        fpath = os.path.join(analysis_path, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as file:
                st.download_button(f"⬇ Download {fname}", data=file, file_name=fname)
