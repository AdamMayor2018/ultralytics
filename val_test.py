from ultralytics import YOLO
import pdb

# Load a model


# model_name = ['train_v8s_baseline', 'yolo8_s5_modify_DCN_100', 'yolo8_s45_modify_DCN', 'yolo8_s345_modify_DCN', 'yolo8_s2345_modify_DCN']a

model_name = ['yolo8_DFPN_ep_60_smootihing_0.2']


# model_name = ['yolo8_s2345_modify_DCN']
model_name = ['yolo8_merge_size_768']


mode = 'test'


# txt_file = mode+'.txt'

# file = open(txt_file, 'w')
for cur_name in model_name:
    # file.write(cur_name + '\n')
    print('processing : {}'.format(cur_name))
    model = YOLO("runs/detect/{}/weights/best.pt".format(cur_name))  # build from YAML and transfer weights
    # Train the model
    results = model.val(data="etl_test.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
#     for key,val in zip(results.keys, results.maps.tolist()):
#         file.write('{} : {} \n'.format(key, val))
# file.close()