import streamlit as st
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

ANALYSIS_PATH = "analysis"
DATASET_PATH = "datasets"

# -------------------------------------------------
# LOAD USERS
# -------------------------------------------------
def load_users():
    with open("users.yaml") as file:
        return yaml.load(file, Loader=SafeLoader)

users_data = load_users()

# -------------------------------------------------
# SESSION INIT
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

def login(username, password):
    if username in users_data["users"]:
        if users_data["users"][username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
if not st.session_state.logged_in:

    st.title("🔐 YOLOv8 Enterprise Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.rerun()
        else:
            st.error("Invalid Credentials")

# -------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------
else:

    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.title("🚀 YOLOv8 Enterprise Dashboard")

    # =========================================================
    # 1️⃣ ACCURACY HISTORY GRAPH
    # =========================================================
    st.header("📈 Accuracy History")

    csv_path = os.path.join(ANALYSIS_PATH, "results.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "metrics/mAP50(B)" in df.columns:
            accuracy = df["metrics/mAP50(B)"]

            fig = plt.figure()
            plt.plot(accuracy)
            plt.xlabel("Epoch")
            plt.ylabel("mAP50")
            plt.title("Accuracy Progress Over Epochs")
            st.pyplot(fig)

        else:
            st.warning("mAP column not found in CSV")

    else:
        st.warning("results.csv not found")

    st.markdown("---")

    # =========================================================
    # 2️⃣ DATASET IMAGE VIEWER
    # =========================================================
    st.header("🗂 Dataset Viewer")

    if os.path.exists(DATASET_PATH):

        image_files = [f for f in os.listdir(DATASET_PATH)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        st.write(f"Total Images: {len(image_files)}")

        selected_image = st.selectbox("Select Image", image_files)

        if selected_image:
            image_path = os.path.join(DATASET_PATH, selected_image)
            image = Image.open(image_path)
            st.image(image, use_container_width=True)

    else:
        st.warning("datasets folder not found")

    st.markdown("---")

    # =========================================================
    # 3️⃣ FAP CALCULATION (False Alarm Percentage)
    # =========================================================
    st.header("📊 FAP Calculation")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
            precision = df["metrics/precision(B)"].iloc[-1]
            recall = df["metrics/recall(B)"].iloc[-1]

            # Simple derived FAP logic (example approximation)
            fap = (1 - precision) * 100

            st.metric("Final Precision", f"{precision:.3f}")
            st.metric("Final Recall", f"{recall:.3f}")
            st.metric("Estimated FAP (%)", f"{fap:.2f}%")

        else:
            st.warning("Required columns not found in CSV")

    st.markdown("---")

    st.success("Dashboard Running Successfully ✅")
