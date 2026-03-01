import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")

st.title("YOLOv8 Object Detection App")

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

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # -------------------------
    # Detection
    # -------------------------

    with st.spinner("Running Detection..."):
        results = model(temp_path)
        result = results[0]

    # Show detected image
    annotated_image = result.plot()
    st.image(annotated_image, caption="Detected Image", use_column_width=True)

    # -------------------------
    # Detection Table
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
        st.subheader("Detection Results")
        st.dataframe(df)

    else:
        st.warning("No objects detected.")

    # Remove temp file
    os.remove(temp_path)
