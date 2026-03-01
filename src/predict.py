import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    try:
        # 🔥 Correct Model Path
        model_path = Path("analysis/best.pt")

        if not model_path.exists():
            print("❌ Model file not found at:", model_path)
            sys.exit(1)

        # 🔥 Test Image Path (Change if needed)
        source_path = Path("test.jpeg")

        if not source_path.exists():
            print("❌ Test image not found:", source_path)
            sys.exit(1)

        # Load model
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
