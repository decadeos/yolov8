from pathlib import Path
import time

import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from sahi import AutoDetectionModel


import cv2

vid_path = Path(__file__).parent / "test_videos/ground_cam_goods_rec.mp4"

cap = cv2.VideoCapture(vid_path)
backSub = cv2.createBackgroundSubtractorMOG2(
    history=120,
    varThreshold=16,
    detectShadows=False
)

kernel_size = 9
offset = 40

t = time.time()

while True:

    ret, frame = cap.read()
    if ret == False: break

    h_frame, w_frame = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    fg_mask = backSub.apply(blurred, 0.002)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 5
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    
    frame_out = frame.copy()

    if len(valid_contours) >= 2:
        continue
    else:
        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)

            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(w_frame, x + w + offset)
            y2 = min(h_frame, y + h + offset)

            frame_out = cv2.rectangle(frame_out, (x1, y1), (x+w+offset, y+h+offset), (0, 0, 200), 1)

            cropped_frame = frame[y1:y2, x1:x2].copy()
            cv2.imshow('jujuju', cropped_frame)
 
    cv2.imshow('lalala', frame_out)



    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'): break

cap.release()
cv2.destroyAllWindows()