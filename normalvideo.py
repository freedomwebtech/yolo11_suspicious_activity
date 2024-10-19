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
video_writer = cv2.VideoWriter("object_blurring_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1020, 600))

# Variable to store the user-selected track_id for blurring
selected_track_id = None
blur_mode = False  # Mode to indicate whether we are blurring a specific track_id

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    # Resize both frames for display
    frame = cv2.resize(frame, (1020, 600))
    frame1 = frame.copy()  # Copy of the original frame for observation

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

            # Blur only the selected track_id if blur_mode is enabled
            if blur_mode and track_id == selected_track_id:
                # Apply blur to the ROI for the selected object
                blur_obj = cv2.blur(roi, (blur_ratio, blur_ratio))
                # Place the blurred ROI back into the original frame
                frame[y1:y2, x1:x2] = blur_obj
            else:
                # Draw rectangle and add labels for all visible objects
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                

    # Write the processed frame to the video writer
    video_writer.write(frame)

    # Display both frames: 'RGB' for processed frame and 'FRAME' for observation frame
    cv2.imshow("RGB", frame)  # Processed frame
    cv2.imshow("FRAME", frame1)  # Observation frame (unprocessed)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Ask user for the track_id when 's' is pressed
        try:
            selected_track_id = int(input("Enter the track_id to blur: "))
            blur_mode = True  # Enable blur mode
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            selected_track_id = None  # Reset the selection if input is invalid
    elif key == ord('n'):
        # Reset and make all objects visible (disable blurring)
        blur_mode = False
        selected_track_id = None

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()
