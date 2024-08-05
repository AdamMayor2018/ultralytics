from ultralytics import YOLO

# # Load a model
model = YOLO("yolov8s_DFPN_columnv2.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
results = model.train(data="/data/cx/datasets/yolo-exp-etl-data/data.yaml", epochs=100, imgsz=640, name='yolo8_DFPN_columnV2',)