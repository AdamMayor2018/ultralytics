_base_ = './yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py'

data_root = '/data/dataset/private/yolo-exp-etl-data/'
class_name = ('Inkiness', 'Vitium', 'Crease', 'defaced', 'Patch', 'Signature')

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

num_last_epochs = 5

max_epochs = 40
train_batch_size_per_gpu = 12
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)))

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

param_scheduler = [
    dict(
        # use quadratic formula to warm up 3 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 35 epoch
        type='CosineAnnealingLR',
        eta_min=_base_.base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

_base_.custom_hooks[0].num_last_epochs = num_last_epochs

val_evaluator = dict(ann_file=data_root + 'coco_format_label/val.json')
test_evaluator = dict(ann_file=data_root + 'coco_format_label/test.json')


test_dataloader = val_dataloader
test_evaluator = val_evaluator



default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(max_epochs=max_epochs, val_interval=20)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
