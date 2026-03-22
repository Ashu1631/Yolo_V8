**Ashu YOLO AI**
A Streamlit app for running YOLO object detection on images, videos, and live webcam feeds.

What it does

Upload & detect — drop in an image or MP4 and see bounding boxes with confidence scores
Live webcam — real-time detection streamed in the browser via WebRTC
Model comparison — run best.pt vs yolov8n.pt side by side on the same input
Evaluation dashboard — view training curves, confusion matrix, mAP and loss charts from a results.csv
Benchmarking — 10 charts comparing accuracy, latency, and throughput between models

**Login: admin / ashu@123**

Setup
pip install streamlit ultralytics supervision streamlit-webrtc plotly pandas opencv-python

project/
├── app.py
├── best.pt
├── yolov8n.pt
├── datasets/  
└── analysis/
    ├── results.csv
    ├── confusion_matrix.png
    ├── BoxF1_curve.png
    └── BoxPR_curve.png
The analysis/ folder and .pt files are not included — bring your own trained weights and YOLO training outputs.

