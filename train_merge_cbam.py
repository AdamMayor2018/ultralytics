from ultralytics import YOLO

# # Load a model
model = YOLO("yolov8s_merge_cbam.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
results = model.train(data="/data/cx/datasets/yolo-exp-etl-data/data.yaml", epochs=100, imgsz=640, name='yolo8_merge_dfpn_cbam_s45')