_backend_args = None
auto_scale_lr = dict(base_batch_size=256)
backend_args = None
base_lr = 0.004
batch_augments_interval = 1
center_radius = 2.5
class_name = (
    'Inkiness',
    'Vitium',
    'Crease',
    'defaced',
    'Patch',
    'Signature',
)
custom_hooks = [
    dict(
        new_train_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='mmdet.Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='mmdet.Pad'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='mmdet.FilterAnnotations'),
            dict(type='mmdet.PackDetInputs'),
        ],
        num_last_epochs=5,
        priority=48,
        type='YOLOXModeSwitchHook'),
    dict(priority=48, type='mmdet.SyncNormHook'),
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
]
data_root = '/data/dataset/private/yolo-exp-etl-data/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 0.33
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
ema_momentum = 0.0002
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
launcher = 'none'
load_from = 'work_dirs/yolox_s_fast_1xb12-40e-rtmdet-hyp_cat_et1_data_detection/best_coco_bbox_mAP_epoch_40.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_aux_weight = 1.0
loss_bbox_weight = 5.0
loss_cls_weight = 1.0
loss_obj_weight = 1.0
max_epochs = 40
max_keep_ckpts = 3
metainfo = dict(
    classes=(
        'Inkiness',
        'Vitium',
        'Crease',
        'defaced',
        'Patch',
        'Signature',
    ),
    palette=[
        (
            20,
            220,
            60,
        ),
    ])
mixup_ratio_range = (
    0.8,
    1.6,
)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        frozen_stages=4,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        spp_kernal_sizes=(
            5,
            9,
            13,
        ),
        type='YOLOXCSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            feat_channels=256,
            featmap_strides=(
                8,
                16,
                32,
            ),
            in_channels=256,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=6,
            stacked_convs=2,
            type='YOLOXHeadModule',
            use_depthwise=False,
            widen_factor=0.5),
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='mmdet.IoULoss'),
        loss_bbox_aux=dict(
            loss_weight=1.0, reduction='sum', type='mmdet.L1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        type='YOLOXHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                interval=1,
                random_size_range=(
                    480,
                    800,
                ),
                size_divisor=32,
                type='YOLOXBatchSyncRandomResize'),
        ],
        pad_size_divisor=32,
        type='YOLOv5DetDataPreprocessor'),
    init_cfg=dict(
        a=2.23606797749979,
        distribution='uniform',
        layer='Conv2d',
        mode='fan_in',
        nonlinearity='leaky_relu',
        type='Kaiming'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_channels=256,
        type='YOLOXPAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        score_thr=0.001,
        yolox_style=True),
    train_cfg=dict(
        assigner=dict(
            center_radius=2.5,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            type='mmdet.SimOTAAssigner')),
    type='YOLODetector',
    use_syncbn=False)
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.65, type='nms'),
    score_thr=0.001,
    yolox_style=True)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 6
num_last_epochs = 5
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=3,
        type='mmdet.QuadraticWarmupLR'),
    dict(
        T_max=35,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=35,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
    dict(begin=35, by_epoch=True, end=40, factor=1, type='ConstantLR'),
]
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
random_affine_scaling_ratio_range = (
    0.1,
    2,
)
resume = False
save_epoch_intervals = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco_format_label/test.json',
        data_prefix=dict(img='test/images'),
        data_root='/data/dataset/private/yolo-exp-etl-data/',
        metainfo=dict(
            classes=(
                'Inkiness',
                'Vitium',
                'Crease',
                'defaced',
                'Patch',
                'Signature',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='mmdet.Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='mmdet.Pad'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/data/dataset/private/yolo-exp-etl-data/coco_format_label/test.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='mmdet.Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='mmdet.Pad'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'annotations/instances_train2017.json'
train_batch_size_per_gpu = 12
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=40,
    type='EpochBasedTrainLoop',
    val_interval=20)
train_data_prefix = 'train2017/'
train_dataloader = dict(
    batch_size=12,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='coco_format_label/train.json',
        data_prefix=dict(img='train/images'),
        data_root='/data/dataset/private/yolo-exp-etl-data/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'Inkiness',
                'Vitium',
                'Crease',
                'defaced',
                'Patch',
                'Signature',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                type='Mosaic'),
            dict(
                border=(
                    -320,
                    -320,
                ),
                scaling_ratio_range=(
                    0.1,
                    2,
                ),
                type='mmdet.RandomAffine'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                ratio_range=(
                    0.8,
                    1.6,
                ),
                type='YOLOXMixUp'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                keep_empty=False,
                min_gt_bbox_wh=(
                    1,
                    1,
                ),
                type='mmdet.FilterAnnotations'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ),
                type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 4
train_pipeline_stage1 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        scaling_ratio_range=(
            0.1,
            2,
        ),
        type='mmdet.RandomAffine'),
    dict(
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        ratio_range=(
            0.8,
            1.6,
        ),
        type='YOLOXMixUp'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        keep_empty=False,
        min_gt_bbox_wh=(
            1,
            1,
        ),
        type='mmdet.FilterAnnotations'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='mmdet.Resize'),
    dict(
        pad_to_square=True,
        pad_val=dict(img=(
            114.0,
            114.0,
            114.0,
        )),
        type='mmdet.Pad'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        keep_empty=False,
        min_gt_bbox_wh=(
            1,
            1,
        ),
        type='mmdet.FilterAnnotations'),
    dict(type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='mmdet.Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='mmdet.Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='mmdet.Resize'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(
                    pad_to_square=True,
                    pad_val=dict(img=(
                        114.0,
                        114.0,
                        114.0,
                    )),
                    type='mmdet.Pad'),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'annotations/instances_val2017.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val2017/'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='coco_format_label/val.json',
        data_prefix=dict(img='val/images'),
        data_root='/data/dataset/private/yolo-exp-etl-data/',
        metainfo=dict(
            classes=(
                'Inkiness',
                'Vitium',
                'Crease',
                'defaced',
                'Patch',
                'Signature',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='mmdet.Resize'),
            dict(
                pad_to_square=True,
                pad_val=dict(img=(
                    114.0,
                    114.0,
                    114.0,
                )),
                type='mmdet.Pad'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/data/dataset/private/yolo-exp-etl-data/coco_format_label/val.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 0.5
work_dir = './work_dirs/yolox_s_fast_1xb12-40e-rtmdet-hyp_cat_et1_data_detection'
