from collections import defaultdict
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model with GPU
model = YOLO("model/yolov8n.pt")

# Open the video file
video_path = "video/VR_serv_11.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video filename without extension
video_filename = os.path.splitext(os.path.basename(video_path))[0]

# Get the video's properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'{video_filename}_tracked.mp4', fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize frame counter
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Check if there are any detections
        if results[0].boxes is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            # Update the track history
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track_history[track_id].append((float(x), float(y)))  # x, y center point

                # Draw the tracking lines
                points = np.array(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Write the frame with tracks to the output video
        out.write(frame)

        # Increment the frame counter
        frame_count += 1

        # Print the processing percentage (overwrite previous line)
        print(f"\rProcessing: {frame_count / total_frames * 100:.2f}%", end="")

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Create a blank image to draw the final tracks on
track_image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw the final tracks
for track_id, track in track_history.items():
    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(track_image, [points], isClosed=False, color=(230, 230, 230), thickness=2)

# Save the track image
track_image_path = f"{video_filename}_tracks.png"
cv2.imwrite(track_image_path, track_image)

# Save the track history to a file
track_file_path = f"{video_filename}.track"
with open(track_file_path, "w") as f:
    for track_id, track in track_history.items():
        for point in track:
            f.write(f"{track_id},{point[0]},{point[1]}\n")

print(f"\nTrack history saved to {track_file_path}")
print(f"Track image saved to {track_image_path}")
print(f"Track video saved to {video_filename}_tracked.mp4")