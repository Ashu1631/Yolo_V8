import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch


def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️ GPU not detected. Using CPU.")
        return "cpu"


def main():
    try:
        # Detect device automatically
        device = check_gpu()

        # Paths
        data_path = Path("data/data.yaml")

        if not data_path.exists():
            print("❌ data.yaml not found inside data/ folder.")
            sys.exit(1)

        # Load pretrained YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Train model
        model.train(
            data=str(data_path),
            epochs=100,
            imgsz=640,
            batch=16,
            device=device,
            project="runs",
            name="vehicle_detection",
            exist_ok=True
        )

        print("\n✅ Training completed successfully!")
        print("📁 Check: runs/vehicle_detection/weights/best.pt")

    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
