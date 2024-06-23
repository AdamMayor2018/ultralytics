from ultralytics import YOLO
import pdb

# Load a model


# model_name = ['train_v8s_baseline', 'yolo8_s5_modify_DCN_100', 'yolo8_s45_modify_DCN', 'yolo8_s345_modify_DCN', 'yolo8_s2345_modify_DCN']

model_name = ['yolo8_merge_ep150_label_smoothing_0.1']
# model_name =['yolo8_s45_modify_DCN']

# model_name = ['yolo8_s2345_modify_DCN']


mode = 'test'


txt_file = mode+'.txt'

# file = open(txt_file, 'w')
for cur_name in model_name:
    print('processing : {}'.format(cur_name))
    # file.write(cur_name + '\n')
    model = YOLO("runs/detect/{}/weights/best.pt".format(cur_name))  # build from YAML and transfer weights
    # Train the model
    results = model.val(data="etl.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
#     for key,val in zip(results.keys, results.maps.tolist()):
#         file.write('{} : {} \n'.format(key, val))
# file.close()