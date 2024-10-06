import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np

# Define the path to the video file
video_path = "susp4.mp4"

def detect_shoplifting(video_path):
    # Load YOLOv8 model (replace with the actual path to your YOLOv8 model)
    model_yolo = YOLO('yolo11n-pose.pt')

    # Load the trained XGBoost model (replace with the actual path to your XGBoost model)
    model = xgb.Booster()
    model.load_model('trained_model.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Generate a unique output path for the processed video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = fr"path_to_your_output_folder\{video_name}_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_tot = 0
    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break  # Stop the loop if no frame is read

        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy  # Bounding box coordinates
            conf = r.boxes.conf.tolist()  # Confidence levels
            keypoints = r.keypoints.xyn.tolist()  # Keypoints for human pose

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:  # Threshold for confidence score
                    x1, y1, x2, y2 = box.tolist()

                    # Prepare data for XGBoost prediction
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    # Convert the data to a DataFrame
                    df = pd.DataFrame(data, index=[0])

                    # Prepare data for XGBoost prediction
                    dmatrix = xgb.DMatrix(df)

                    # Make prediction using the XGBoost model
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)
                    print(f'Prediction: {binary_predictions}')

                    # Annotate the frame based on prediction (0 = Suspicious, 1 = Normal)
                    if binary_predictions == 0:  # Suspicious
                        conf_text = f'Suspicious ({conf[index]:.2f})'
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 7, 58), 2)
                        cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 7, 58), 2)
                    else:  # Normal
                        conf_text = f'Normal ({conf[index]:.2f})'
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (57, 255, 20), 2)
                        cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (57, 255, 20), 2)

        # Show the annotated frame in a window
        cv2.imshow('Frame', annotated_frame)

        # Write the frame to the output video
        out.write(annotated_frame)
        frame_tot += 1
        print('Processed Frame:', frame_tot)

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()

    # Close all OpenCV windows after processing is complete
    cv2.destroyAllWindows()

# Call the function with the video path
detect_shoplifting(video_path)
