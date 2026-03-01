from ultralytics import YOLO
import argparse

def run_inference(weights, source):

    model = YOLO(weights)

    results = model.predict(
        source=source,
        save=True,
        conf=0.25
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)

    args = parser.parse_args()

    run_inference(args.weights, args.source)
