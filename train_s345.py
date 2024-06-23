from ultralytics import YOLO

# Load a model
model = YOLO("yolov8_s345.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='yolo8_s345_modify_DCN')

