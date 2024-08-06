from ultralytics import YOLO
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='yolov8s_DFPN_map_v3.yaml', type=str,
                        help='name of the experiment')
    parser.add_argument('--mode', default='test', type=str, choices=['val', 'test'])
    parser.add_argument('--data', default='/data/cx/datasets/yolo-exp-etl-data/data.yaml', type=str, help='data yaml path')
    args = parser.parse_args()
    print('processing : {}'.format(args.exp_name))
    model = YOLO("/data/cx/aigc/stable-diffusion-webui/runs/detect/yolov8s_DFPN_map_v3.yaml/weights/best.pt".format(args.exp_name))
    #model = YOLO("runs/detect/{}/weights/best.pt".format(args.exp_name))
    results = model.val(data=args.data, imgsz=640, batch=16, conf=0.25, iou=0.6, device="1", split=args.mode)