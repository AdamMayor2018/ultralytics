import json
import os
import cv2
from tqdm import tqdm

txt_path = '/cfs_train_data/wangyuxuan/dataset/dabuguo_dataset/xindabuguo_val.txt'
# 输入文件夹包含 YOLO 格式标注文件的路径
yolo_annotation_folder = '/cfs_train_data/wangyuxuan/dataset/dabuguo_dataset/xindabuguo/xindabuguo_label'
img_base_path = '/cfs_train_data/wangyuxuan/dataset/dabuguo_dataset/xindabuguo/pull_data'

# COCO 格式输出文件
coco_output_file = '/cfs_train_data/yanzong/xudong/raw_data/detction/dabuguo/val.json'


def get_img_path(path):
    return [x.path for x in os.scandir(path) if (x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".jpeg")  or x.name.endswith(".JPGG") or x.name.endswith(".PNG") )]




data_dict = {
    "train" : "/data/dataset/private/yolo-exp-etl-data/train/images",
    "val" : "/data/dataset/private/yolo-exp-etl-data/val/images",
    "test" : "/data/dataset/private/yolo-exp-etl-data/test/images"
}



def coco_inint():
    # 映射类别名称和ID
    category_map = {0: "Inkiness",
                    1: "Vitium",
                    2: "Crease",
                    3: "defaced",
                    4: "Patch",
                    5: "Signature",
                    }





    # COCO 数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for class_id, class_name in category_map.items():
        category_info = {
            "id": class_id+ 1,
            "name": class_name,
            "supercategory": "object"
        }

        coco_data["categories"].append(category_info)
    return coco_data





for cur_mode, img_base_path in data_dict.items():
    base_path = "/".join(img_base_path.split('/')[:-2])
    out_json_label_path = os.path.join(base_path, 'coco_format_label')
    os.makedirs(out_json_label_path, exist_ok=True)
    coco_output_file = os.path.join(out_json_label_path, cur_mode+'.json')
    yolo_annotation_folder = os.path.join(base_path, cur_mode, 'labels')
    img_list = get_img_path(img_base_path)

    coco_data = coco_inint()
    for cur_img_path in tqdm(img_list):
        img_name = os.path.basename(cur_img_path)
        txt_name ='.'.join(img_name.split('.')[:-1])+'.txt'

        

        yolo_annotation_file = os.path.join(yolo_annotation_folder, txt_name)
        if not os.path.exists(yolo_annotation_file):
            print('label path not exist : {}'.format(yolo_annotation_file))

        img = cv2.imread(cur_img_path)
        cur_img_heigh, cur_img_width, _ = img.shape

        # 获取图像信息
        image_info = {
            "id": len(coco_data["images"]) + 1,
            "file_name": os.path.basename(cur_img_path),
            "height": cur_img_heigh,  # 替换为实际图像高度
            "width": cur_img_width    # 替换为实际图像宽度
        }

        coco_data["images"].append(image_info)


        with open(yolo_annotation_file, 'r') as file:
            lines = file.readlines()
            
            # 遍历 YOLO 格式标注数据
            for idx,line in enumerate(lines):
                if idx == len(lines) - 1:
                    continue
                parts = line.strip().split(' ')
                            
                # 获取类别信息
                class_id = parts[0]
                
                # 获取边界框信息
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                x_min = int((x_center - width / 2) * image_info["width"])
                y_min = int((y_center - height / 2) * image_info["height"])
                x_max = int((x_center + width / 2) * image_info["width"])
                y_max = int((y_center + height / 2) * image_info["height"])
                
                # 构建 COCO 格式标注
                annotation_info = {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_info["id"],
                    "category_id": int(class_id) + 1,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                    "ignore": 0
                }
                coco_data["annotations"].append(annotation_info)

    # 保存为 COCO 格式 JSON 文件
    with open(coco_output_file, 'w') as outfile:
        json.dump(coco_data, outfile)
