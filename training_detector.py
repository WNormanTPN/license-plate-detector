from ultralytics import YOLO

# Load YOLOv8s pre-trained model
model = YOLO('./models/yolo11n.pt')

# Train YOLOv8s on custom data
model.train(data='./data.yaml', epochs=10, batch=32, imgsz=512)

# Save and export the model
model.save('./models/yolo11n_custom.pt')