
cfg_path=configs/yolov6/yolov6_s_fast_1xb12-40e_cat_et1_data_detection.py
model_path=work_dirs/yolov6_s_fast_1xb12-40e_cat_et1_data_detection/best_coco_bbox_mAP_epoch_35.pth
python tools/test.py $cfg_path \
                     $model_path \
                     --cfg-options test_evaluator.classwise=True

