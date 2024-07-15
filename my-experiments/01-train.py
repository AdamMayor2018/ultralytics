# -- coding: utf-8 --
# @Time : 2024/2/4 15:23
# @Author : caoxiang
# @File : 01-train.py
# @Software: PyCharm
# @Description: yolov8基础测试 操作指令链接 https://docs.ultralytics.com/modes/train/#arguments
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.yaml').load("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/data/cx/datasets/yolo-exp-etl-data/data.yaml', imgsz=640, device='1', project="/data/cx/ysp-2024/yolov8-exp/runs/yolo-exp-v2", name='yolov8s-attentionconcat-eca')
