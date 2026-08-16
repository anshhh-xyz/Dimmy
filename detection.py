import cv2 as cv
from main_predict_yolo import predict_frame

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = predict_frame(frame)

    cv.imshow('YOLO + CNN Object Detection', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
