from ultralytics import YOLO

# # Load a model
model = YOLO("yolov8_merge.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
results = model.train(data="etl.yaml", epochs=150, imgsz=640, label_smoothing=0.1,name='yolo8_merge_ep150_label_smoothing_0.1')