import cv2
from ultralytics import YOLO

# Load pre-trained YOLO model
model = YOLO('models/yolo11n_custom.pt') 

# Input video
input_video_path = "VID_20230506_163430.mp4"
output_video_path = "output_video.mp4"

# Open video
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Draw detections on the frame
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        class_id = int(result.cls[0])  # Class ID
        confidence = float(result.conf[0])  # Confidence score

        # Draw bounding box and label
        label = f"{model.names[class_id]} {confidence:.2f}"
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write frame to output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
