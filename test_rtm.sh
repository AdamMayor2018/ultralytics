
cfg_path=configs/rtmdet/rtmdet_tiny_fast_1xb12-40e_cat_et1_data_detection.py
# model_path=work_dirs/yolov5_s-v61_fast_1xb12-40e_cat_et1_data_detection/epoch_10.pth
model_path=/data/code/mmyolo/work_dirs/rtmdet_tiny_fast_1xb12-40e_cat_et1_data_detection/best_coco_bbox_mAP_epoch_40.pth
python tools/test.py $cfg_path \
                     $model_path \
                     --cfg-options test_evaluator.classwise=True

