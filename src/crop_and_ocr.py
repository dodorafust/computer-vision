# === File: crop_and_ocr.py ===
import easyocr
import cv2
import numpy as np
import re

# Khởi tạo EasyOCR một lần duy nhất để tối ưu hiệu suất
reader = easyocr.Reader(['en', 'vi'])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    equalized = cv2.equalizeHist(sharpened)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    return blurred

def extract_license_plate_text(plate_img):
    try:
        processed_img = preprocess_image(plate_img)
        results = reader.readtext(processed_img)
        if results:
            valid_results = [r for r in results if r[2] > 0.6]
            if valid_results:
                text = max(valid_results, key=lambda x: x[2])[1]
                text = text.upper().replace(" ", "").strip()
                valid_chars = re.findall(r'[A-Z0-9]', text)
                return ''.join(valid_chars)
    except Exception as e:
        print(f"OCR Error: {e}")
    return None
