from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from functions import saving_prediction_video

# CONSTANTS
prediction_video_input = input("If you want to save the video - type, else pass...")
MODEL_PATH = "Models/9_nov_10_epochs.pt"
CONFIDENCE = 0.6

# webcam/video
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the weights
model = YOLO(MODEL_PATH)

# object classes
classNames = ["Fake", "Real"]

# prediction video
if prediction_video_input:
    video_prediction = saving_prediction_video(cap)

while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            # convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # class name
            cls = int(box.cls[0])

            if conf > CONFIDENCE:
                # color
                if classNames[cls] == 'Real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                # draw rectangle and class name
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)
                # writing frames for prediction video
                if prediction_video_input:
                    video_prediction.write(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
