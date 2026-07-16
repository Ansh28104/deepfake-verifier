import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=5):
    if int(frame_rate) <= 0:
        raise ValueError("frame_rate must be a positive integer")

    frame_interval = int(frame_rate)
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Resize to model input size
            resized = cv2.resize(frame, (380, 380))  # EfficientNet-B4
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), resized)
            saved_count += 1
        frame_count += 1
    cap.release()
    return saved_count