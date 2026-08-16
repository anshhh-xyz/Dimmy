import os
import cv2 as cv
from ultralytics import YOLO
from main_predict_cnn import classify_sign

model_weights = r"C:\ME\Project-Detection\runs\detect\idd_yolov11s_local\weights\best.pt"
model = YOLO(model_weights)


def predict_frame(frame):
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()

        if "traffic sign" in label or "traffic light" in label:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            sign_crop = frame[y1:y2, x1:x2]
            sign_name, conf = classify_sign(sign_crop)

            print(f"• Sign detected: {sign_name} ({conf:.1f}%)")

            cv.putText(
                annotated_frame,
                f"{sign_name} ({conf:.0f}%)",
                (x1, max(20, y1 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv.LINE_AA
            )

    return annotated_frame
