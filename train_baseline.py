from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8_s2345s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='s2345_baseline')




# # Load a model
# model = YOLO("yolov8_s345s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='s345_baseline')


# # Load a model
# model = YOLO("yolov8_s45s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='s45_baseline')

# # Load a model
# model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='yolo8_s2345_modify_DCN')



# Load a model
# model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=100, imgsz=640,name='train_v8s_baseline')


# # Load a model
# model = YOLO("yolov8_DFPN.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="etl.yaml", epochs=60, imgsz=640, label_smoothing=0.2,name='yolo8_DFPN_ep_60_smootihing_0.2')

# Load a model
model = YOLO("yolov8_merge.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="etl.yaml", epochs=150, imgsz=640, label_smoothing=0.1, name='yolo8_merge_ep200_label_smoothing_0.1')