from ultralytics import YOLO
import argparse
import torch


def train_model(data_path, model_size, epochs, imgsz, batch, device):

    print("🚀 Starting YOLOv8 Training...")
    print(f"📂 Dataset: {data_path}")
    print(f"🧠 Model: {model_size}")
    print(f"📏 Image Size: {imgsz}")
    print(f"📦 Batch Size: {batch}")
    print(f"🔁 Epochs: {epochs}")
    print(f"🖥 Device: {device}")
    print("-" * 50)

    # Load model
    model = YOLO(model_size)

    # Train
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs",
        name="vehicle_detection",
        exist_ok=True
    )

    print("✅ Training Completed Successfully!")
    print("📁 Check weights at: runs/vehicle_detection/weights/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train YOLOv8 Model")

    parser.add_argument("--data", type=str, default="data/data.yaml",
                        help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model (yolov8n.pt, yolov8s.pt etc.)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0 for GPU, cpu for CPU (auto if not set)")

    args = parser.parse_args()

    # Auto-detect GPU if device not specified
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_model(
        data_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device
    )
