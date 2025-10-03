import cv2
import os
from ultralytics import YOLO

def extract_best_frame(video_path, model_path='model/best.pt', output_dir='output_frames'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    best_frame = None
    max_box_area = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)[0]

        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            # Pick the frame with largest visible tractor
            if conf > 0.5 and area > max_box_area:
                max_box_area = area
                best_frame = frame.copy()

    cap.release()

    if best_frame is not None:
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.basename(video_path).split('.')[0]
        save_path = os.path.join(output_dir, f"{video_name}_best.jpg")
        cv2.imwrite(save_path, best_frame)
        return save_path
    else:
        return None
