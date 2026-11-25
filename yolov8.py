from pathlib import Path

from ultralytics import YOLO
import yaml

CONFIG_PATH = Path(__file__).parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())

MODEL_NAME = str(dict(config.get("train")).get("name"))
EPOCH = int(dict(config.get("train")).get("epochs"))
BATCH = int(dict(config.get("train")).get("batch_size"))
DST_DIR = Path(config["output_dir"])   

PROJECT_DIR = Path(__file__).parent / "runs" / "train"

def main():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=f"{DST_DIR}/data.yaml",
        epochs=EPOCH,
        batch=BATCH,
        imgsz=640,
        project=PROJECT_DIR,
        name=MODEL_NAME,
        exist_ok=True,
        patience=50,
        resume=True
    )

if __name__ == "__main__":
    main()