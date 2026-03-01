import streamlit as st
import yaml
from yaml.loader import SafeLoader

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="YOLOv8 Enterprise Dashboard",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------
# Load Users
# ----------------------------
def load_users():
    with open("users.yaml") as file:
        return yaml.load(file, Loader=SafeLoader)

users_data = load_users()

# ----------------------------
# Session State Init
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None


# ----------------------------
# Login Function
# ----------------------------
def login(username, password):
    if username in users_data["users"]:
        if users_data["users"][username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
    return False


# ----------------------------
# Logout Function
# ----------------------------
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None


# ----------------------------
# LOGIN PAGE
# ----------------------------
if not st.session_state.logged_in:

    st.title("🔐 YOLOv8 Enterprise Dashboard Login")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if login(username, password):
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ----------------------------
# MAIN DASHBOARD
# ----------------------------
else:

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.title("🚀 YOLOv8 Enterprise Dashboard")

    st.markdown("---")

    # Example Dashboard Layout
    col1, col2, col3 = st.columns(3)

    col1.metric("Model Version", "YOLOv8n")
    col2.metric("Accuracy", "92.4%")
    col3.metric("Total Predictions", "1,248")

    st.markdown("---")

    st.subheader("📂 Upload Image for Detection")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.info("YOLO detection logic yahan add karo.")

    st.markdown("---")
    st.subheader("📊 Evaluation Section")
    st.write("Confusion Matrix, Accuracy Graph, Model Comparison etc. yahan integrate kar sakte ho.")
