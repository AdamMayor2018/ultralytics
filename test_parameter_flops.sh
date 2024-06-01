
cfg_path=configs/rtmdet/rtmdet_tiny_fast_1xb12-40e_cat_et1_data_detection.py
cfg_path=configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat_et1_data_detection.py
# cfg_path=configs/yolov6/yolov6_s_fast_1xb12-40e_cat_et1_data_detection.py
# cfg_path=configs/yolov8/yolov8_s_fast_1xb12-40e_cat_et1_data_detection.py
# cfg_path=configs/yolox/yolox_s_fast_1xb12-40e-rtmdet-hyp_cat_et1_data_detection.py


python tools/analysis_tools/get_flops.py $cfg_path 
