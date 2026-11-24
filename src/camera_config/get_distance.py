import cv2, math
import numpy as np

_prev_dark_pixels = -1

def get_distance(dark_pixels, K, threshold_dark=12):

    if K < 0 or dark_pixels < threshold_dark:
        dist = -1
    else:
        dist = K / math.sqrt(dark_pixels) # метры

    return round(dist, 1)


def find_pixels_bbox(im0, xyxy, threshold_k=2):

    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    weight = x2 - x1   
    height = y2 - y1
    total_pixels = weight * height

    roi = im0[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('output.jpg', binary_image)

    dark_pixels = np.sum(binary_image == 0).item()

    if _prev_dark_pixels != -1 and _prev_dark_pixels != 0:

        k = dark_pixels / _prev_dark_pixels

        if k > threshold_k:
            dark_pixels = _prev_dark_pixels
        else: 
            _prev_dark_pixels = dark_pixels

    return total_pixels, dark_pixels, binary_image


def show_binary(im0, binary_image, total_pixels, dark_pixels, margin=10):

    mask_h, mask_w = binary_image.shape[:2]
    _, frame_w = im0.shape[:2]
    
    x_start = frame_w - margin - mask_w
    y_start = margin
    x_end = frame_w - margin
    y_end = margin + mask_h

    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    im0[y_start:y_end, x_start:x_end] = binary_image

    label_total = get_label("total", total_pixels)
    label_dark = get_label("dark", dark_pixels)

    draw_text_with_outline(im0, label_total, margin*2, margin*3)
    draw_text_with_outline(im0, label_dark, margin*2, margin*6)

    return im0

def get_K(Z, dark_pixels, threshold_dark=12):

    # дистанция до объекта K (const для каждой камеры)
    # if dark_pixels < threshold_dark:
    #     K = -1
    # else: 
    #     K = Z * dark_pixels -- пусть будет так
    # Где Z - расстояние до объекта, которое мы вычислили
    # K_i = [метр * пиксель]

    if dark_pixels < threshold_dark: K = -1
    else: 
        K = Z * math.sqrt(dark_pixels)

    return K



# utils

def get_label(name, value): 
    return f"{name}: {value}"

def draw_text_with_outline(image, text, x, y, font_scale=0.5, color=(255, 255, 255), thickness=1, outline_color=(0, 0, 0), outline_thickness=2):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
