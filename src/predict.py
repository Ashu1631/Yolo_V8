from ultralytics import YOLO
import argparse
import os


def run_inference(weights, source, conf):

    print("🔎 Starting Inference...")
    print(f"📦 Weights: {weights}")
    print(f"🎯 Source: {source}")
    print(f"📊 Confidence Threshold: {conf}")
    print("-" * 50)

    if not os.path.exists(weights):
        raise FileNotFoundError(f"❌ Weights file not found: {weights}")

    # Load trained model
    model = YOLO(weights)

    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        show=False,
        project="outputs",
        name="predictions",
        exist_ok=True
    )

    print("✅ Inference Completed!")
    print("📁 Outputs saved in: outputs/predictions/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (best.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, video file, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")

    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        conf=args.conf
    )
