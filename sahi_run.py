from pathlib import Path
import os
import yaml
import numpy as np
import cv2
from tqdm import tqdm

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- Конфигурация ---
CONFIG_PATH = Path(__file__).parent / "params.yaml"
config = yaml.safe_load(CONFIG_PATH.read_text())
MODEL_NAME = str(dict(config.get("train")).get("name"))

MASTER_DIR = Path(config["master_dir"])
DST_DIR = Path(config["output_dir"])  

# Пути
BEST_PT = f"runs/train/{MODEL_NAME}/weights/best.pt"
DATA_YAML = f"{DST_DIR}/data.yaml"
TEST_IMG_DIR = f"{DST_DIR}/images/test"
LABELS_DIR = f"{DST_DIR}/labels/test"

CONF = 0.35
IOU_TRES = 0.45

print(f"[YOLOv8 model] {MODEL_NAME}")

def calculate_metrics(predictions, gt_boxes, iou_threshold=IOU_TRES):
    """Честный подсчет TP/FP/FN при фиксированном IoU"""
    tp, fp, fn = 0, 0, 0
    used_gt = [False] * len(gt_boxes)
    
    for pred in predictions:
        # Поддержка обоих форматов: SAHI (pred.bbox) и Ultralytics (xyxy тензор)
        if hasattr(pred, 'bbox'):
            x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        else:
            x1, y1, x2, y2 = map(float, pred[:4])
            
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if used_gt[i]: 
                continue
                
            xx1 = max(x1, gt[0])
            yy1 = max(y1, gt[1])
            xx2 = min(x2, gt[2])
            yy2 = min(y2, gt[3])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            
            area_pred = (x2 - x1) * (y2 - y1)
            area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
            union = area_pred + area_gt - inter
            
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_threshold:
            tp += 1
            used_gt[best_gt_idx] = True
        else:
            fp += 1
            
    fn = len(gt_boxes) - sum(used_gt)
    return tp, fp, fn

def get_gt_boxes(img_path, label_path):
    """Парсинг Ground Truth из txt файла"""
    gt_boxes = []
    frame = cv2.imread(img_path)
    if frame is None: 
        return [], 0, 0
        
    h_img, w_img = frame.shape[:2]
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    px = x * w_img; py = y * h_img
                    pw = w * w_img; ph = h * h_img
                    gt_boxes.append([px - pw/2, py - ph/2, px + pw/2, py + ph/2])
                    
    return gt_boxes, h_img, w_img

def main():
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png'))])
    output_dir = "sahi_vis"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. ЧЕСТНАЯ ВАЛИДАЦИЯ ЧИСТОЙ YOLOv8
    # ==========================================
    print(f"--- [1/2] Запуск честной валидации YOLOv8 (IoU={IOU_TRES}, conf={CONF}) ---")
    model_ultra = YOLO(BEST_PT)
    
    # Используем predict вместо val для идентичных условий с SAHI
    ultra_preds = model_ultra.predict(
        source=TEST_IMG_DIR,
        data=DATA_YAML,
        conf=CONF,
        iou=IOU_TRES,
        save=False,
        verbose=False
    )

    total_tp_yolo, total_fp_yolo, total_fn_yolo = 0, 0, 0
    
    for pred in tqdm(ultra_preds, desc="Evaluating Pure YOLO"):
        img_name = os.path.basename(pred.path)
        label_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        gt_boxes, _, _ = get_gt_boxes(pred.path, label_path)
        
        # Приводим предсказания к единому формату для calculate_metrics
        yolo_preds_formatted = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for box in pred.boxes:
                yolo_preds_formatted.append(box.xyxy[0].cpu().numpy())

        tp, fp, fn = calculate_metrics(yolo_preds_formatted, gt_boxes, iou_threshold=IOU_TRES)
        total_tp_yolo += tp
        total_fp_yolo += fp
        total_fn_yolo += fn

    precision_yolo = total_tp_yolo / (total_tp_yolo + total_fp_yolo) if (total_tp_yolo + total_fp_yolo) > 0 else 0
    recall_yolo = total_tp_yolo / (total_tp_yolo + total_fn_yolo) if (total_tp_yolo + total_fn_yolo) > 0 else 0
    
    print(f"[YOLOv8 Fair] Precision: {precision_yolo:.4f} | Recall: {recall_yolo:.4f}\n")

    # ==========================================
    # 2. SAHI ВАЛИДАЦИЯ
    # ==========================================
    print("--- [2/2] Запуск SAHI inference ---")
    
    model_sahi = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=BEST_PT,
        confidence_threshold=CONF,
        device="cuda:0"
    )

    total_tp_sahi, total_fp_sahi, total_fn_sahi = 0, 0, 0

    for img_name in tqdm(test_images, desc="Processing SAHI"): 
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        gt_boxes, _, _ = get_gt_boxes(img_path, label_path)

        result = get_sliced_prediction(
            image=img_path,
            detection_model=model_sahi,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            postprocess_type="GREEDYNMM",
            verbose=False
        )
        
        tp, fp, fn = calculate_metrics(result.object_prediction_list, gt_boxes, iou_threshold=IOU_TRES)
        total_tp_sahi += tp
        total_fp_sahi += fp
        total_fn_sahi += fn
        
        # Сохранение визуализации
        temp_filename = f"temp_{img_name}"
        result.export_visuals(export_dir=output_dir, file_name=temp_filename)
        vis_path = os.path.join(output_dir, f"{temp_filename}.jpg")
        if not os.path.exists(vis_path):
             vis_path = os.path.join(output_dir, f"{temp_filename}.png")

    precision_sahi = total_tp_sahi / (total_tp_sahi + total_fp_sahi) if (total_tp_sahi + total_fp_sahi) > 0 else 0
    recall_sahi = total_tp_sahi / (total_tp_sahi + total_fn_sahi) if (total_tp_sahi + total_fn_sahi) > 0 else 0
    
    print(f"\n--- ИТОГИ SAHI ---")
    print(f"SAHI Precision: {precision_sahi:.4f}")
    print(f"SAHI Recall:    {recall_sahi:.4f}")

    # ==========================================
    # 3. ДЕЛЬТА
    # ==========================================
    delta_p = precision_sahi - precision_yolo
    delta_r = recall_sahi - recall_yolo
    
    print(f"\n--- DELTA (SAHI - YOLO) ---")
    print(f"Δ Precision: {delta_p:+.4f}")
    print(f"Δ Recall:    {delta_r:+.4f}")
    print(f"Визуализации сохранены в: ./{output_dir}")

if __name__ == "__main__":
    main()