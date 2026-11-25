from pathlib import Path

from ultralytics import YOLO
import yaml

CONFIG_PATH = Path(__file__).parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())

MODEL_NAME = str(dict(config.get("train")).get("name"))
EPOCH = str(dict(config.get("train")).get("epochs"))
BATCH = str(dict(config.get("train")).get("batch_size"))

DST_DIR = Path(config["output_dir"])   

MODEL_PATH = f"runs/train/{MODEL_NAME}/weights/best.pt"

print(MODEL_NAME)
print(DST_DIR)
print(MODEL_PATH)


def validate_model(model_path):
    model = YOLO(model_path)
    results = model.val(data=DST_DIR)
    
    print(f"mAP50-95: {results.box.map}")
    print(f"mAP50: {results.box.map50}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")
    
    return results


if __name__ == "__main__":
    results = validate_model(MODEL_PATH)