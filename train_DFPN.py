from ultralytics import YOLO

# # Load a model
model = YOLO("yolov8_DFPN.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
results = model.train(data="etl.yaml", epochs=60, imgsz=640, label_smoothing=0.2,name='yolo8_DFPN_ep_60_smootihing_0.2')