import shutil
import random
from pathlib import Path
import yaml

from log import *

random.seed(42)

CONFIG_PATH = Path(__file__).parent.parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())

MASTER_DIR = Path(config["master_dir"])       # dataset-master-local
DST_DIR = Path(config["output_dir"])          # dataset-bol
PREFIXES = config.get("prefixes", {})
SPLITS = config.get("splits", {"train": 0.7, "valid": 0.2, "test": 0.1})
OBJECT_BANKS = config.get("object_banks", [])
BG_BANK = config.get("bg_bank", "background")
BG_RATIO = config.get("bg_ratio", 1)
CLASSES = config["classes"]
NUM_CLASSES = len(eval(CLASSES))

DST_DIR_NAME = DST_DIR.name

def detete_pass():

    for bank, _ in PREFIXES.items():
        img_dir = MASTER_DIR / bank / "images"
        lbl_dir = MASTER_DIR / bank / "labels"

        
        for file_img in sorted(img_dir.iterdir()):

            current_filename = Path(file_img).stem

            if file_img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:

                if  current_filename == "classes":
                    logger.info(f"[delete] Файл классов в банке объектов, удален")
                    file_img.unlink(missing_ok=True)
                
                continue

            file_txt = lbl_dir / f"{current_filename}.txt"

            if not file_txt.exists() and file_img.exists() and bank not in BG_BANK:
                logger.info(f"[delete] Файл без разметки в банке объектов {file_txt}")
                file_img.unlink(missing_ok=True)

        for file_txt in sorted(lbl_dir.iterdir()):

            current_filename = Path(file_txt).stem

            file_img = img_dir / f"{current_filename}.png"

            has_image = any(
                (img_dir / f"{current_filename}{ext}").exists() 
                for ext in [".png", ".jpg", ".jpeg"]
            )

            if file_txt.exists() and not has_image and current_filename != "classes":
                logger.info(f"[delete] Разметка без картинки в банке, удалена {file_txt}")
                file_txt.unlink(missing_ok=True)

def rename(digits=4):
    renamed_count = 0

    for bank, prefix in PREFIXES.items():
        img_dir = MASTER_DIR / bank / "images"
        lbl_dir = MASTER_DIR / bank / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        images = sorted([f for f in img_dir.iterdir() 
                         if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

        idx = 0
        temp_renames = []

        for file_img in images:
            current_stem = file_img.stem
            file_txt = lbl_dir / f"{current_stem}.txt"

            idx += 1
            real_idx = f"{idx:0{digits}d}"
            
            tmp_img_name = f"__tmp_{real_idx}{file_img.suffix}"
            tmp_txt_name = f"__tmp_{real_idx}.txt"
            
            temp_renames.append((
                file_img, file_img.parent / tmp_img_name,
                file_txt, file_txt.parent / tmp_txt_name
            ))

        for old_i, new_i, old_t, new_t in temp_renames:
            if old_i.exists(): old_i.rename(new_i)
            if old_t.exists(): old_t.rename(new_t)
  

        final_count = 0
        for _, tmp_i, _, tmp_t in temp_renames:
            if tmp_i.exists():
                final_name = tmp_i.name.replace("__tmp_", prefix)
                tmp_i.rename(tmp_i.parent / final_name)
            if tmp_t.exists():
                final_name = tmp_t.name.replace("__tmp_", prefix)
                tmp_t.rename(tmp_t.parent / final_name)
            final_count += 1
            
        renamed_count += final_count

    print()
    logger.info(f"[rename] Переименовано: {renamed_count}\n")

def build_split():
    """Собирает инкрементальный сплит в dataset-bol"""
    for split in SPLITS:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    obj_pairs = []
    for bank in OBJECT_BANKS:
        img_dir = MASTER_DIR / bank / "images"
        lbl_dir = MASTER_DIR / bank / "labels"

        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            txt = lbl_dir / f"{img.stem}.txt"

            # не включать пары, если индекс в txt больше, чем всего классов
            lines = txt.read_text(encoding="utf-8").splitlines()
            for line in lines:
                parts = line.split()
                current_clas = int(parts[0])
                if current_clas > NUM_CLASSES:
                    continue

            if txt.exists():
                obj_pairs.append((img, txt))

    bg_banks_list = BG_BANK if isinstance(BG_BANK, list) else [BG_BANK]

    all_bg = []
    for bank_name in bg_banks_list:
        bg_img_dir = MASTER_DIR / bank_name / "images"
        if bg_img_dir.exists():
            files = [f for f in bg_img_dir.glob("*") 
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            all_bg.extend(files)
            
    n_bg = int(len(obj_pairs) * BG_RATIO)
    selected_bg = random.sample(all_bg, min(n_bg, len(all_bg)))
    bg_pairs = [(img, None) for img in selected_bg]

    all_data = obj_pairs + bg_pairs

    # Проверяем, что уже есть в dataset-bol
    existing_files = set()
    for split in SPLITS:
        for f in (DST_DIR / "images" / split).glob("*"):
            existing_files.add(f.name)

    new_data = [pair for pair in all_data if pair[0].name not in existing_files]

    if not new_data:
        logger.info("[split] Новых файлов не найдено. Датасет актуален.")
        return

    logger.info(f"[split] Найдено {len(new_data)} новых файлов для распределения")

    random.shuffle(new_data)
    total_new = len(new_data)
    train_end = int(total_new * SPLITS["train"])
    val_end = int(total_new * (SPLITS["train"] + SPLITS["valid"]))

    splits_data = {
        "train": new_data[:train_end],
        "valid": new_data[train_end:val_end],
        "test": new_data[val_end:],
    }

    for split_name, pairs in splits_data.items():
        img_dst = DST_DIR / "images" / split_name
        lbl_dst = DST_DIR / "labels" / split_name
        for img_src, lbl_src in pairs:
            shutil.copy2(img_src, img_dst / img_src.name)
            if lbl_src:
                shutil.copy2(lbl_src, lbl_dst / lbl_src.name)
        logger.info(f"[split] {split_name}: добавлено {len(pairs)} файлов")

    yaml_content = \
f"""path: {DST_DIR}
train: images/train
val: images/valid
test: images/test

nc: {NUM_CLASSES}
names: {CLASSES}
"""
    print(f"\n************\n{yaml_content}************\n")
    (DST_DIR / "data.yaml").write_text(yaml_content)
    logger.info("[split] data.yaml обновлён")


if __name__ == "__main__":

    detete_pass()
    rename()
    build_split()