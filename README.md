🚀 Ashu YOLO Enterprise Pro
Ashu YOLO Enterprise Pro ek industrial-grade Computer Vision dashboard hai jo object detection models (YOLOv8) ko deploy, evaluate aur compare karne ke liye banaya gaya hai. Yeh application researchers aur developers ko real-time inference aur deep analytical insights pradan karti hai.

✨ Key Features
1. 🛡️ Secure Login System
Admin-level authentication (admin / ashu@1234) enterprise usage ko secure rakhne ke liye.

2. 📦 Model Selection & Multi-Model Support
Dynamic Loading: Apne custom .pt files ko runtime par select aur load karein.

Dual Model Support: Ek saath do models (Primary & Secondary) load karne ki suvidha.

3. 🔍 Advanced Detection Hub
Image & Video: Dono formats par high-speed inference.

Real-time Comparison: Do models ki performance ko side-by-side video stream mein compare karein.

Sleek Annotation: Supervision library ka use karke high-quality bounding boxes aur labels.

4. 📹 WebRTC Live Feed
Browser-based webcam detection bina kisi lag ke, firewall bypass ke liye STUN/TURN servers integrated.

5. 📊 Evaluation & Analytics Dashboard
Performance Curves: Loss Curve, Confusion Matrix, Box F1, aur PR Curves ka visual analysis.

CSV Integration: Training logs (results.csv) se metrics ko automatically parse karke KPI metrics dikhata hai.

6. ⚖️ Model Benchmarking (10-Graph Matrix)
Advanced Plotly visualizations:

Latency vs Accuracy Scatter

Precision/Recall Bar & Pie Charts

Throughput Analysis & Heatmaps

🛠️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/yourusername/yolo-enterprise-pro.git
cd yolo-enterprise-pro
Install Dependencies:

Bash
pip install -r requirements.txt
Directory Structure:
Ensure aapke root folder mein ye folders maujood honge:

analysis/ (Loss curves aur results.csv ke liye)

datasets/ (Test images ke liye)

outputs/ (Inference results ke liye)

Run the App:

Bash
streamlit run app.py
📂 Project Architecture
Plaintext
├── analysis/               # Training results and performance graphs
├── datasets/               # Sample images for testing
├── outputs/                # Processed images and videos
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── best.pt                 # Your custom YOLO model (place here)
📊 Technical Stack
Inference Engine: Ultralytics YOLOv8

Frontend: Streamlit

Live Streaming: Streamlit-WebRTC & PyAV

Visualization: Plotly & Supervision

Image Processing: OpenCV

🤝 Contributing
Agar aap is project ko behtar banana chahte hain, toh:

Repository ko Fork karein.

Naya Branch banayein (git checkout -b feature/AmazingFeature).

Apne changes Commit karein (git commit -m 'Add some AmazingFeature').

Branch ko Push karein (git push origin feature/AmazingFeature).

Ek Pull Request open karein.

📜 License
Distributed under the MIT License. See LICENSE for more information.

Developed with ❤️ by [Ashu]
