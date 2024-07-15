from ultralytics import YOLO
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='yolo8_merge', type=str,
                        help='name of the experiment')
    parser.add_argument('--mode', default='val', type=str, choices=['val', 'test'])
    parser.add_argument('--data', default='etl.yaml', type=str, help='data yaml path')
    args = parser.parse_args()
    print('processing : {}'.format(args.exp_name))
    model = YOLO("runs/detect/{}/weights/best.pt".format(args.exp_name).format(args.exp_name))
    #model = YOLO("runs/detect/{}/weights/best.pt".format(args.exp_name))
    results = model.val(data=args.data, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0", split=args.mode)