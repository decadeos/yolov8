import os
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())

MASTER_DIR = Path(config["master_dir"])
PREFIXES = config.get("prefixes", {})

NAME = "red_balls"

mapa = [['0', '2'], ['1', '2']]

lbl_dir = MASTER_DIR / NAME / "labels"

for filename in sorted(lbl_dir.iterdir()):

    if filename.name == "classes.txt": continue

    with open(filename, 'r', encoding='utf-8') as f:

        lines = f.readlines()
        new_lines = []

        if len(lines) > 1:
            print(filename, len(lines))

        for line in lines:

            parts = line.strip().split(' ')

            for i in range(len(mapa)):

                if parts and parts[0] == mapa[i][0]:
                    parts[0] = mapa[i][1]

            new_lines.append(' '.join(parts) + '\n')

        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
