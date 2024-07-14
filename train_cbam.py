from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-cbam.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/data/cx/datasets/yolo-exp-etl-data/data.yaml", epochs=100, imgsz=640, name='train_v8s_cbam')

