import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="YOLOv8 Advanced Detection App", layout="wide")

st.title("YOLOv8 Object Detection Dashboard")

# -------------------------
# Model Selection
# -------------------------

model_option = st.selectbox(
    "Select Model",
    ["yolov8n.pt", "best.pt"]
)

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_option)

# -------------------------
# Image Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # -------------------------
    # Detection
    # -------------------------

    with st.spinner("Running Detection..."):
        results = model(temp_path)
        result = results[0]

    annotated_image = result.plot()
    st.image(annotated_image, caption="Detected Image", use_column_width=True)

    # -------------------------
    # Extract Results
    # -------------------------

    if result.boxes:

        names = result.names
        data = []

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            data.append({
                "Object": names[cls],
                "Confidence": round(conf, 2)
            })

        df = pd.DataFrame(data)

        st.subheader("Detection Results Table")
        st.dataframe(df)

        # -------------------------
        # Object Count Graph
        # -------------------------

        st.subheader("Object Count Graph")
        count_df = df["Object"].value_counts()

        fig1, ax1 = plt.subplots()
        count_df.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Object")
        st.pyplot(fig1)

        # -------------------------
        # Confidence Graph
        # -------------------------

        st.subheader("Confidence Score Graph")

        fig2, ax2 = plt.subplots()
        ax2.bar(df["Object"], df["Confidence"])
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        st.pyplot(fig2)

    else:
        st.error("No objects detected!")

        # -------------------------
        # Failure Cause Section
        # -------------------------

        st.subheader("Possible Failure Causes")

        st.markdown("""
        - Low image quality
        - Object too small
        - Object not in trained classes
        - Poor lighting
        - Wrong model selected
        """)

    os.remove(temp_path)
