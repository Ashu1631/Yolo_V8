import streamlit as st
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from streamlit_autorefresh import st_autorefresh

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

ANALYSIS_PATH = "analysis"
DATASET_PATH = "datasets"

# =====================================================
# AUTO REFRESH OPTION
# =====================================================
if st.sidebar.checkbox("Auto Refresh (30 sec)"):
    st_autorefresh(interval=30000, key="datarefresh")

# =====================================================
# LOAD USERS
# =====================================================
def load_users():
    if os.path.exists("users.yaml"):
        with open("users.yaml") as file:
            return yaml.load(file, Loader=SafeLoader)
    return {"users": {}}

users_data = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

def login(username, password):
    if username in users_data.get("users", {}):
        if users_data["users"][username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# =====================================================
# LOGIN PAGE
# =====================================================
if not st.session_state.logged_in:

    st.title("🔐 YOLOv8 Enterprise Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.rerun()
        else:
            st.error("Invalid Credentials")

# =====================================================
# MAIN DASHBOARD
# =====================================================
else:

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.title("🚀 YOLOv8 Production Dashboard")
    st.markdown("---")

    # =====================================================
    # ACCURACY HISTORY GRAPH
    # =====================================================
    st.header("📈 Accuracy History")

    csv_path = os.path.join(ANALYSIS_PATH, "results.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "metrics/mAP50(B)" in df.columns:
            fig = plt.figure()
            plt.plot(df["metrics/mAP50(B)"])
            plt.xlabel("Epoch")
            plt.ylabel("mAP50")
            plt.title("Accuracy Over Epochs")
            st.pyplot(fig)

        if "metrics/precision(B)" in df.columns:
            precision = df["metrics/precision(B)"].iloc[-1]
            fap = (1 - precision) * 100

            col1, col2 = st.columns(2)
            col1.metric("Final Precision", f"{precision:.4f}")
            col2.metric("False Alarm % (FAP)", f"{fap:.2f}%")
    else:
        st.warning("results.csv not found in analysis folder")

    st.markdown("---")

    # =====================================================
    # DATASET GRID VIEW + PAGINATION
    # =====================================================
    st.header("🗂 Dataset Viewer")

    if os.path.exists(DATASET_PATH):

        images = sorted([
            f for f in os.listdir(DATASET_PATH)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        total_images = len(images)
        st.write(f"Total Images: {total_images}")

        if total_images > 0:

            per_page = 12
            total_pages = max(1, (total_images + per_page - 1) // per_page)

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )

            start = (page - 1) * per_page
            end = min(start + per_page, total_images)

            cols = st.columns(4)

            for idx, img in enumerate(images[start:end]):
                col = cols[idx % 4]
                image_path = os.path.join(DATASET_PATH, img)
                col.image(image_path, use_container_width=True)

            st.info(f"Showing {start+1} - {end} of {total_images}")

        else:
            st.warning("No images found in datasets folder")

    else:
        st.warning("datasets folder not found")

    st.markdown("---")

    # =====================================================
    # MODEL SELECTION
    # =====================================================
    st.header("📊 Model Selection")

    if os.path.exists(ANALYSIS_PATH):
        model_files = [
            f for f in os.listdir(ANALYSIS_PATH)
            if f.endswith(".pt")
        ]

        if model_files:
            selected_model = st.selectbox(
                "Select Model",
                model_files
            )
            st.success(f"Selected Model: {selected_model}")
        else:
            st.warning("No .pt model found inside analysis folder")
    else:
        st.warning("analysis folder not found")

    st.markdown("---")

    # =====================================================
    # YOLO INFERENCE ON DATASET PAGE
    # =====================================================
    st.header("🧠 Run YOLO Inference")

    if 'selected_model' in locals():
        if st.button("Run Inference on Current Page Images"):

            try:
                model_path = os.path.join(ANALYSIS_PATH, selected_model)
                model = YOLO(model_path)

                for img in images[start:end]:
                    image_path = os.path.join(DATASET_PATH, img)
                    results = model(image_path)
                    result_img = results[0].plot()
                    st.image(result_img, caption=img, use_container_width=True)

            except Exception:
                st.error("Model inference failed")

    st.markdown("---")
    st.success("Dashboard Running Successfully ✅")
