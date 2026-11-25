from pathlib import Path
import cv2
import numpy as np

DATASETS_PATH = Path(__file__).parent / "datasets/test"
FULL_DATA = DATASETS_PATH / "full/images"
CROP_DATA = DATASETS_PATH / "crop/images"

def get_single_adaptive_crop(mask_clean, frame_shape, target_size=1000, padding=40):
    """
    Находит ОДИН адаптивный кроп для всей зоны движения
    """
    h_frame, w_frame = frame_shape[:2]
    
    # 1. Находим все точки движения
    coords = np.column_stack(np.where(mask_clean > 0))
    
    if len(coords) == 0:
        return None
        
    # 2. Строим ОДИН объединяющий bbox по всем точкам
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 3. Добавляем padding
    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)
    x2 = min(w_frame, x_max + padding)
    y2 = min(h_frame, y_max + padding)
    
    # 4. Приводим к квадрату target_size × target_size
    # Если зона меньше 640 — центрируем и дополняем черным
    # Если больше 640 — берем центр (или ресайзим, если критично)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    half = target_size // 2
    final_x1 = max(0, cx - half)
    final_y1 = max(0, cy - half)
    final_x2 = min(w_frame, cx + half)
    final_y2 = min(h_frame, cy + half)
    
    # Центрируем относительно кадра, если вылезли за границы
    if final_x2 - final_x1 < target_size:
        diff = target_size - (final_x2 - final_x1)
        if final_x1 == 0:
            final_x2 = min(w_frame, final_x2 + diff)
        elif final_x2 == w_frame:
            final_x1 = max(0, final_x1 - diff)
            
    if final_y2 - final_y1 < target_size:
        diff = target_size - (final_y2 - final_y1)
        if final_y1 == 0:
            final_y2 = min(h_frame, final_y2 + diff)
        elif final_y2 == h_frame:
            final_y1 = max(0, final_y1 - diff)

    return (final_x1, final_y1, final_x2, final_y2)

vid_path = Path(__file__).parent / "test_videos/ground_cam_goods_rec.mp4"

cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2(
    history=120,
    varThreshold=16,
    detectShadows=False
)

kernel_size = 9
offset = 40

while True:

    ret, frame = cap.read()
    if ret == False: break

    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    fg_mask = backSub.apply(blurred, 0.002)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    crop_coords = get_single_adaptive_crop(mask_clean, frame.shape, target_size=640, padding=40)

    frame_bbox = frame.copy()

    if crop_coords is not None:
        x1, y1, x2, y2 = crop_coords
        crop = frame[y1:y2, x1:x2].copy()
        
        cv2.rectangle(frame_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    cv2.imshow('frame_bbox', frame_bbox)
    cv2.imshow('crop', crop)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'): break

cap.release()
cv2.destroyAllWindows()