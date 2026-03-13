import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from ultralytics import YOLO
import supervision as sv

# --- Minimal Supervision Helper ---
def apply_supervision(image, results):
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    
    labels = [f"{results[0].names[class_id]} {conf:.2f}" 
              for class_id, conf in zip(detections.class_id, detections.confidence)]
    
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

st.title("🚀 Standalone Webcam Test")

# Load model once at the start
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Use nano model for faster testing

model = load_model()

class TestProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        results = model(img, verbose=False)
        
        # Annotation
        annotated_img = apply_supervision(img, results)
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

webrtc_streamer(
    key="test-webcam",
    video_processor_factory=TestProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
