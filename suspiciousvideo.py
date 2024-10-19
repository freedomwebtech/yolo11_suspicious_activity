import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLOv8 model
model = YOLO("yolo11s.pt")
names = model.model.names

# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('susp1.mp4')
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

blur_ratio = 50

# Ensure frame size matches when writing the video
frame_size = (1020, 600)

# Use a codec that matches the file format
video_writer = cv2.VideoWriter("object_blurring_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)

# Variable to store the user-selected track_id
selected_track_id = None
blur_all = True  # Start with all objects blurred

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    # Resize frame to the correct dimensions
    frame = cv2.resize(frame, frame_size)
    frame1 = frame.copy()

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, classes=0)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x1, y1, x2, y2), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box

            # Extract the region of interest (ROI) from the frame
            roi = frame[y1:y2, x1:x2]

            # Apply blur if blur_all is enabled or if the track_id is not the selected one
            if blur_all or (selected_track_id is not None and track_id != selected_track_id):
                # Apply blur to the ROI
                blur_obj = cv2.blur(roi, (blur_ratio, blur_ratio))
                # Place the blurred ROI back into the original frame
                frame[y1:y2, x1:x2] = blur_obj

            # Draw rectangle around the object (whether blurred or not)
            color = (0, 255, 0) if track_id == selected_track_id else (0, 0, 255)  # Green for unblurred, red for blurred
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'Track ID: {track_id}', (x1, y2 + 20), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

    # Display the result
    cv2.imshow("RGB", frame)
    cv2.imshow("FRAME", frame1)

    # Write the processed frame to the video writer
    video_writer.write(frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Ask user for the track_id to keep unblurred when 's' is pressed
        try:
            selected_track_id = int(input("Enter the track_id to keep unblurred: "))
            blur_all = False  # Disable full blur mode
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            selected_track_id = None  # Reset the selection if input is invalid
    elif key == ord('n'):
        # When 'n' is pressed, blur all objects again
        blur_all = True
        selected_track_id = None  # Clear the selected track_id when blurring all objects

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()
