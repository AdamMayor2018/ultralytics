from ultralytics import YOLO
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='train_v8s_baseline', type=str, help='opt train_v8s_baseline, yolo8_s5_modify_DCN_100, yolo8_s45_modify_DCN, yolo8_s345_modify_DCN, yolo8_s2345_modify_DCN, yolo8_merge_ep150_label_smoothing_0.1, yolo8_s45_modify_DCN')
parser.add_argument('--mode', default='val', type=str, choices=['val', 'test'])



if __name__ == '__main__':
    args = parser.parse_args()
    print('processing : {}'.format(args.exp_name))
    model = YOLO("runs/detect/{}/weights/best.pt".format(args.exp_name)) 
    results = model.val(data="etl.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", split=args.mode)