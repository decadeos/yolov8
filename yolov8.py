from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference on the sample image
results = model("https://ultralytics.com/images/bus.jpg")

# Display the annotated results
results[0].show()