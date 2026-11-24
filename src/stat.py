# import pandas
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate
import yaml
from log import *

WIDTH = 1280
HEIGHT = 720

CONFIG_PATH = Path(__file__).parent.parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())
DATASET_FOLDER = Path(config["master_dir"])

OBJECT_BANKS = config.get("object_banks", [])
BG_BANK = config.get("bg_bank", "background")

folders = [x for x in DATASET_FOLDER.iterdir() if x.is_dir()]
big_count = sum(1 for _ in DATASET_FOLDER.rglob("*.jpg")) + \
            sum(1 for _ in DATASET_FOLDER.rglob("*.png")) + \
            sum(1 for _ in DATASET_FOLDER.rglob("*.jpeg"))

bg_banks_list = BG_BANK if isinstance(BG_BANK, list) else [BG_BANK]

folders = [DATASET_FOLDER / bank for bank in OBJECT_BANKS] + \
          [DATASET_FOLDER / bank for bank in bg_banks_list]
rows = []

for folder in folders:
    
    if folder.name in bg_banks_list:
        bg_count = sum(1 for _ in (folder / "images").glob("*"))

        rows.append({
            "folder": folder.name,
            "count": bg_count,
            "%": f"{bg_count / big_count * 100:03.2f}",
            "mean S": "-",
            "min S": "-",
            "max S": "-",
        })
        continue

    count = 0
    areas = []

    labels_path = folder / "labels"

    for item in sorted(labels_path.iterdir()):
        if item.name == "classes.txt" or item.suffix != ".txt":
            continue

        count += 1

        with open(item, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            parts = content.split()
            w_norm, h_norm = float(parts[3]), float(parts[4])
            s = w_norm * WIDTH * h_norm * HEIGHT
            areas.append(s)

        except IndexError:
            print("Найден файл без разметки: ", item)

    rows.append({
        "folder": folder.name,
        "count": count,
        "%": f"{count / big_count * 100:03.2f}",
        "mean S": int(np.mean(areas)),
        "min S": int(min(areas)),
        "max S": int(max(areas)),
    })

rows = pd.DataFrame(rows)

print(tabulate(rows, headers=list(rows), tablefmt="rounded_grid"), "\n")

