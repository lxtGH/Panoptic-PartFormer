num_stages = 3
conv_kernel_size = 1
_base_ = [
    './baseline_r50.py',
]

model = dict(
    rpn_head=dict(
        localization_fam=dict(
            type='UperNetAlignHead',
            in_channels=[256, 256, 256, 256],
            out_channels=256),
        localization_fpn=dict(
            num_aux_convs=1,
        ),
    ),
# swin backbone
    backbone=dict(
        _delete_=True,
        type='SwinTransformerDIY',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
       ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    # iterative kernel
    roi_head=dict(
        mask_head=[
            dict(
                thing_part_stuff_update=False,
                thing_stuff_part_update=False,
                thing_part_stuff_attention=False,
                thing_stuff_part_attention=False,
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

evaluation = dict(metric=['panoptic', 'part'])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPartAnnotations', with_bbox=True, with_mask=True, with_seg=True, with_part=True),
    dict(
        type='PartResize', img_scale=[(512, 1024), (2048, 4096)], multiscale_mode='range', keep_ratio=True),
    dict(type='PartRandomCrop', crop_size=(1024, 2048)),
    dict(type='PartRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PartPad', size_divisor=32),
    dict(type='PartDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_part']),
]


dataset_type = 'CityscapesPanopticPartDataset'
data_root = 'data/cityscapes/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=dict(
                ins_ann=data_root + 'annotations/instancesonly_filtered_gtFine_train.json',
                panoptic_ann=data_root + 'annotations/cityscapes_panoptic_train.json',
            ),
            img_prefix=data_root + 'leftImg8bit/train/',
            seg_prefix=data_root + 'gtFine/train',
            part_prefix=data_root + 'gtFinePanopticParts/train/',
            panoptic_part_eval_config=dict(
                eval_spec_path="panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml",
                basepath_gt="data/cityscapes/gtFinePanopticParts/val",
                basepath_pred="work_dirs/part_res",
                images_json="data/cityscapes/annotations/val_images.json",
                save_dir="work_dirs/part_res/save"
            ),
            pipeline=train_pipeline)),
)