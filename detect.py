import cv2
import time
import os
import argparse
from ultralytics import YOLO
import pyttsx3
from datetime import datetime

# ---------------------------
# Parse Arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="0",
                    help="Source: 0 for webcam, path to image/video file")
args = parser.parse_args()

# Convert "0" string to int for webcam
source = 0 if args.source == "0" else args.source

# ---------------------------
# Initialize voice engine
# ---------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ---------------------------
# Load YOLOv8 model
# ---------------------------
model = YOLO("yolov8n.pt")
TARGET_CLASSES = ["person", "car"]

# ---------------------------
# Load confidence threshold
# ---------------------------
try:
    with open("config.txt", "r") as f:
        conf_threshold = float(f.read().strip())
except:
    conf_threshold = 0.25  # default

# ---------------------------
# Logging setup
# ---------------------------
log_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_detection(label, confidence):
    with open(log_filename, "a") as f:
        f.write(f"{datetime.now()} - Detected: {label} ({confidence:.2f})\n")

# ---------------------------
# Prepare snapshot folder
# ---------------------------
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# ---------------------------
# Capture Source
# ---------------------------
cap = cv2.VideoCapture(source)
frame_width, frame_height = 640, 480
cap.set(3, frame_width)
cap.set(4, frame_height)

detected_set = set()
prev_time = time.time()
last_saved = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] No frame received. Exiting...")
        break

    results = model(frame)[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if confidence < conf_threshold or label not in TARGET_CLASSES:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Voice alert (only first time for each label)
        if label not in detected_set:
            engine.say(f"{label} detected")
            engine.runAndWait()
            detected_set.add(label)

        # Log detection
        log_detection(label, confidence)

    # Save snapshot every 1 second
    curr_time = time.time()
    if curr_time - last_saved >= 1:  # 1 second interval
        filename = f"snapshots/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved snapshot: {filename}")
        last_saved = curr_time

    # FPS calculation
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show window (only for webcam or video)
    if isinstance(source, int) or str(source).endswith((".mp4", ".avi")):
        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # If single image, save and exit
        filename = f"snapshots/detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved snapshot from image: {filename}")
        break

cap.release()
cv2.destroyAllWindows()
