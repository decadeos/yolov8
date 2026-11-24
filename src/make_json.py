
import ast
import json
import os
from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())

MODEL_NAME = str(dict(config.get("train")).get("name"))
CLASSES = ast.literal_eval(config["classes"])


OUTPUT_DIR = Path(__file__).parent.parent / f"runs/train/{MODEL_NAME}/output/"
RK_ANCORS_FILE = OUTPUT_DIR / "RK_anchors.txt"
JSON_FILE = OUTPUT_DIR / "yolo_config.json"

RK_anchors = list()

with open(RK_ANCORS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        RK_anchors.append(ast.literal_eval(line))

os.remove(RK_ANCORS_FILE)


anchors_str = ',\n        '.join(json.dumps(a) for a in RK_anchors)

json_content = f'''{{
    "model_path": "{MODEL_NAME}.rknn",
    "labels": {json.dumps(CLASSES, ensure_ascii=False)},
    "anchors": [
        {anchors_str}
    ]
}}'''

with open(JSON_FILE, "w", encoding="utf-8") as f:
    f.write(json_content + '\n')