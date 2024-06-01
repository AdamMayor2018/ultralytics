
cfg_path=configs/yolov6/yolov6_s_fast_1xb12-40e_cat_et1_data_detection.py
model_path=work_dirs/yolov6_s_fast_1xb12-40e_cat_et1_data_detection/best_coco_bbox_mAP_epoch_40.pth
# rlaunch --gpu=1 --cpu=4 --memory=16000 --positive-tags=p40 -- 
python tools/train.py $cfg_path 