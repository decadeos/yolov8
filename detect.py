from pathlib import Path

from ultralytics import YOLO
import yaml

# Загрузка твоей модели

# --- Конфигурация ---
CONFIG_PATH = Path(__file__).parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())
MODEL_NAME = str(dict(config.get("train")).get("name"))

# Пути
BEST_PT = f"/home/ohrimenkoed/Документы/eva/2027/yolov8/runs/train/{MODEL_NAME}/weights/best.pt"
DATA_YAML = "datasets/dataset-ball/data.yaml"
TEST_IMG_DIR = "datasets/dataset-ball/images/valid"
LABELS_DIR = "datasets/dataset-ball/labels/valid"
model = YOLO("/home/ohrimenkoed/Документы/eva/2027/yolov8/runs/train/v11_test/weights/best.pt")

# Запуск детекции
results = model.predict(
    source="datasets/dataset-ball/images/valid",
    save=True,
    show=False,
    conf=0.25,
    project="/home/ohrimenkoed/Документы/eva/2027/yolov8/runs/detect",
    name="predictions"
)

