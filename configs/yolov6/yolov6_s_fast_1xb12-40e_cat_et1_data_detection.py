_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

data_root = '/data/dataset/private/yolo-exp-etl-data/'
class_name = ('Inkiness', 'Vitium', 'Crease', 'defaced', 'Patch', 'Signature')

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4
num_last_epochs = 5

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco_format_label/train.json',
        data_prefix=dict(img='train/images')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='coco_format_label/val.json',
        data_prefix=dict(img='val/images')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='coco_format_label/test.json',
        data_prefix=dict(img='test/images')))

val_evaluator = dict(ann_file=data_root + 'coco_format_label/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_format_label/test.json')


# test_dataloader = val_dataloader
# test_evaluator = val_evaluator


_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
