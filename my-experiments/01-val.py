#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:44
# @Author  : 作者名
# @File    : 01-val.py.py
# @Description  :
from ultralytics import YOLO
model = YOLO("/data/cx/aigc/stable-diffusion-webui/runs/detect/train19/weights/best.pt")  # build from YAML and transfer weights
print(model)
# Train the model
results = model.val(data="/data/cx/datasets/yolo-exp-etl-data/data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
for key,val in zip(results.keys, results.maps.tolist()):
    print('{} : {} \n'.format(key, val))