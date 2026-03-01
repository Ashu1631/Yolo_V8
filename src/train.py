import sys
import torch
from ultralytics import YOLO
from roboflow import Roboflow


def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️ GPU not detected. Using CPU.")
        return "cpu"


def download_dataset():
    print("⬇️ Downloading dataset from Roboflow...")

    rf = Roboflow(api_key="A828xM1sAEtjHMh5WHGh")
    project = rf.workspace("ashishs-workspace-nho2b").project("traffic-detection-yc4eg")
    version = project.version(1)
    dataset = version.download("yolov8")

    print("✅ Dataset downloaded successfully!")
    return dataset.location + "/data.yaml"


def main():
    try:
        device = check_gpu()

        # Download dataset
        data_path = download_dataset()

        # Load pretrained YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Train model
        model.train(
            data=data_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=device,
            project="runs",
            name="vehicle_detection",
            exist_ok=True
        )

        print("\n🎉 Training Completed Successfully!")
        print("📁 Model saved at: runs/vehicle_detection/weights/best.pt")

    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
