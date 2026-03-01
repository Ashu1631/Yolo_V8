from ultralytics import YOLO
import argparse

def train_model(data_path, model_size, epochs, imgsz, batch):

    model = YOLO(model_size)

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="data/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()

    train_model(args.data, args.model, args.epochs, args.imgsz, args.batch)
