import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone
# Initialize the YOLO model and video capture
model = YOLO('yolo11s-pose.pt')
cap = cv2.VideoCapture('vid5.mp4')

count=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))
    count += 1
    if count % 3 != 0:
        continue
    
    result=model(frame,show=True)


    # Exit on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
