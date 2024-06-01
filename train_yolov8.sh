
cfg_path=configs/yolov8/yolov8_s_fast_1xb12-40e_cat_et1_data_detection.py
# model_path=work_dirs/yolov5_s-v61_fast_1xb12-40e_cat_et1_data_detection/epoch_10.pth
# rlaunch --gpu=1 --cpu=4 --memory=16000 --positive-tags=p40 -- 
python tools/train.py $cfg_path 