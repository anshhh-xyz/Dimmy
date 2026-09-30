"""Standalone webcam demo (no web server). Press 'q' to quit.

For the web app run:  python backend/server.py
"""
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

config.add_backend_paths()
from main_predict_yolo import predict_frame  # noqa: E402


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = predict_frame(frame)

            cv.imshow("YOLO + CNN Object Detection", frame)
            if cv.waitKey(1) == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
