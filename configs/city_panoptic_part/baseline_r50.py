num_stages = 3
num_proposals = 100
conv_kernel_size = 1

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/cityscapes_panoptic_part.py',
    '../_base_/models/partnet_s3_r50_fpn_panoptic.py',
    '../_base_/default_runtime.py'
]

model = dict(
    cityscapes=True,
    num_thing_classes=8,
    num_stuff_classes=11,
    num_part_classes=23,
    # init kernel
    rpn_head=dict(
        num_classes=19,
        num_thing_classes=8,
        num_stuff_classes=11,
        num_part_classes=23,
        cat_stuff_mask=True,
        num_inst_convs=1,
        num_seg_convs=1,
        num_part_cnovs=1
    ),
    # iterative kernel
    roi_head=dict(
        type='PartKernelIterHead',
        eval_spec_path='./panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml',
        output_dir='./work_dirs/part_res/',
        cityscapes=True,
        merge_joint=True,
        num_thing_classes=8,
        num_stuff_classes=11,
        num_part_classes=23,
        mask_head=[
            dict(
                type='PartKernelUpdateHead',
                num_classes=42,
                num_thing_classes=8,
                num_stuff_classes=11,
                num_part_classes=23,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=conv_kernel_size,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0)
            ) for _ in range(num_stages)
        ]),
)

custom_imports = dict(
    imports=[
        'models.panoptic_part',
        'models.losses',
        'datasets',
        'swin.swin_transformer'
    ],
    allow_failed_imports=False)

load_from = "./pretrain/knet_r50_city.pth"
evaluation = dict(metric=['panoptic', 'part'])

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)
checkpoint_config = dict(interval=4)
