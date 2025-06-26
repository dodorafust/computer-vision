import cv2
import torch
import pandas as pd
import difflib
import datetime
import time
import os
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
from crop_and_ocr import extract_license_plate_text

# === Load YOLO model ===
model = YOLO("../models/best.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# === Load dữ liệu người dùng ===
def load_user_data():
    if not os.path.exists("user_data.csv"):
        return pd.DataFrame(columns=["plate", "name", "id"]), {}
    df = pd.read_csv("user_data.csv")
    user_dict = {row['plate']: {"name": row['name'], "id": row['id']} for _, row in df.iterrows()}
    return df, user_dict

user_df, user_dict = load_user_data()

# === Biến toàn cục ===
current_plate = None
current_user = None
history_file = "history.csv"
has_written_header = os.path.exists(history_file)
last_detect_time = 0
DETECTION_INTERVAL = 0.5
register_window = None

# === Hàm tiện ích ===
def find_closest_plate(plate_text, known_plates):
    matches = difflib.get_close_matches(plate_text, known_plates, n=1, cutoff=0.6)
    return matches[0] if matches else None

def resize_plate(img, height=100):
    h, w = img.shape[:2]
    new_w = int((height / h) * w)
    return cv2.resize(img, (new_w, height))

def log_history(plate, name, uid, status):
    global has_written_header
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[plate, name, uid, status, time_str]],
                      columns=["plate", "name", "id", "status", "time"])
    df.to_csv(history_file, mode='a', index=False, header=not has_written_header)
    has_written_header = True

def update_labels(name="Unknown", uid="Unknown", plate="Unknown"):
    name_label.config(text=f"Tên: {name}")
    id_label.config(text=f"Mã số: {uid}")
    plate_label.config(text=f"Biển số: {plate}")

def reset_info():
    global current_plate, current_user
    current_plate = None
    current_user = None
    update_labels()

def allow_entry():
    if current_plate:
        name = current_user["name"] if current_user else "Unknown"
        uid = current_user["id"] if current_user else "Unknown"
        log_history(current_plate, name, uid, "Cho vào")
        reset_info()

def deny_entry():
    if current_plate:
        name = current_user["name"] if current_user else "Unknown"
        uid = current_user["id"] if current_user else "Unknown"
        log_history(current_plate, name, uid, "Từ chối")
        reset_info()

def clear_on_focus(event):
    widget = event.widget
    if widget.get() in ["Nhập biển số", "Nhập tên", "Nhập mã số"]:
        widget.delete(0, END)

def open_register_window():
    global register_window, entry_plate, entry_name, entry_id
    if register_window is not None:
        return
    register_window = Toplevel(root)
    register_window.title("Đăng ký người mới")
    register_window.geometry("300x250")

    Label(register_window, text="🔒 ĐĂNG KÝ NGƯỜI MỚI", font=("Arial", 14, "bold")).pack(pady=10)

    entry_plate = Entry(register_window, font=("Arial", 12))
    entry_name = Entry(register_window, font=("Arial", 12))
    entry_id = Entry(register_window, font=("Arial", 12))

    entry_plate.insert(0, "Nhập biển số")
    entry_name.insert(0, "Nhập tên")
    entry_id.insert(0, "Nhập mã số")

    entry_plate.bind("<FocusIn>", clear_on_focus)
    entry_name.bind("<FocusIn>", clear_on_focus)
    entry_id.bind("<FocusIn>", clear_on_focus)

    entry_plate.pack(pady=5, fill=X, padx=20)
    entry_name.pack(pady=5, fill=X, padx=20)
    entry_id.pack(pady=5, fill=X, padx=20)
    Button(register_window, text="➕ ĐĂNG KÝ", font=("Arial", 12), bg="blue", fg="white", command=lambda: register_user(current_plate)).pack(pady=10)

    register_window.protocol("WM_DELETE_WINDOW", close_register_window)

def close_register_window():
    global register_window
    if register_window:
        register_window.destroy()
        register_window = None

def register_user(plate_text):
    global user_dict, user_df
    name = entry_name.get().strip()
    uid = entry_id.get().strip()
    new_plate = entry_plate.get().strip().upper()

    if not (new_plate and name and uid):
        print("❌ Thiếu thông tin để đăng ký!")
        return

    if new_plate in user_dict:
        print("⚠️ Biển số đã tồn tại!")
        return

    new_row = pd.DataFrame([[new_plate, name, uid]], columns=["plate", "name", "id"])
    new_row.to_csv("user_data.csv", mode='a', index=False, header=not os.path.exists("user_data.csv"))

    user_df, user_dict = load_user_data()
    print(f"✅ Đã đăng ký biển số {new_plate} cho {name} - {uid}")
    update_labels(name, uid, new_plate)
    close_register_window()

# === Camera ===
cap = cv2.VideoCapture(0)

def update_frame():
    global current_plate, current_user, last_detect_time
    ret, frame = cap.read()
    if not ret:
        return

    current_time = time.time()
    if current_time - last_detect_time >= DETECTION_INTERVAL:
        results = model.predict(frame, conf=0.5, verbose=False)
        last_detect_time = current_time

        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                h, w, _ = frame.shape
                pad = 20
                x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                x2p, y2p = min(w, x2 + pad), min(h, y2 + pad)
                plate_img = frame[y1p:y2p, x1p:x2p]

                plate_resized = resize_plate(plate_img)
                plate_text = extract_license_plate_text(plate_resized)

                if plate_text and plate_text != current_plate:
                    current_plate = plate_text
                    matched_plate = find_closest_plate(plate_text, user_dict.keys())
                    if matched_plate:
                        current_user = user_dict[matched_plate]
                        update_labels(current_user['name'], current_user['id'], matched_plate)
                    else:
                        current_user = None
                        update_labels("Unknown", "Unknown", plate_text)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    camera_label.imgtk = img_pil
    camera_label.configure(image=img_pil)

    root.after(30, update_frame)

# === GUI ===
root = Tk()
root.title("🚧 HỆ THỐNG NHẬN DIỆN BIỂN SỐ + KIỂM SOÁT CỔNG VÀO")
root.geometry("1000x750")

camera_label = Label(root)
camera_label.pack(side=LEFT, padx=10, pady=10)

info_frame = Frame(root)
info_frame.pack(side=RIGHT, padx=20, pady=20)

Label(info_frame, text="📋 THÔNG TIN XE", font=("Arial", 18, "bold")).pack(pady=10)
name_label = Label(info_frame, text="Tên: ", font=("Arial", 14))
id_label = Label(info_frame, text="Mã số: ", font=("Arial", 14))
plate_label = Label(info_frame, text="Biển số: ", font=("Arial", 14))
name_label.pack(pady=5)
id_label.pack(pady=5)
plate_label.pack(pady=5)

Button(info_frame, text="✅ CHO VÀO CỔNG", font=("Arial", 14), bg="green", fg="white", command=allow_entry).pack(pady=10, fill=X)
Button(info_frame, text="❌ TỪ CHỐI", font=("Arial", 14), bg="red", fg="white", command=deny_entry).pack(pady=10, fill=X)
Button(info_frame, text="➕ ĐĂNG KÝ", font=("Arial", 12), bg="blue", fg="white", command=open_register_window).pack(pady=10, fill=X)

root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
