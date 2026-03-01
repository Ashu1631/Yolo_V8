import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    try:
        # Model path
        model_path = Path("best.pt")

        if not model_path.exists():
            print("❌ best.pt not found in root directory.")
            print("Place best.pt in project root or update path in predict.py")
            sys.exit(1)

        # Input source
        source_path = Path("input.jpg")

        if not source_path.exists():
            print("❌ input.jpg not found in root directory.")
            sys.exit(1)

        # Load trained model
        model = YOLO(str(model_path))

        # Run prediction
        model.predict(
            source=str(source_path),
            imgsz=640,
            conf=0.25,
            save=True,
            project="outputs",
            name="predictions",
            exist_ok=True
        )

        print("\n✅ Prediction completed successfully!")
        print("📁 Check: outputs/predictions/")

    except Exception as e:
        print(f"\n❌ Prediction Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
