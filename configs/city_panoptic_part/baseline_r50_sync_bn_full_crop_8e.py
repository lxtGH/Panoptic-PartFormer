num_stages = 3
num_proposals = 100
conv_kernel_size = 1
dataset_type = 'CityscapesPanopticPartDataset'
data_root = 'data/cityscapes/'

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/cityscapes_panoptic_part.py',
    '../_base_/models/partnet_s3_r50_fpn_panoptic.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='PartNet',
    cityscapes=True,
    num_thing_classes=8,
    num_stuff_classes=11,
    num_part_classes=23,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # init kernel
    rpn_head=dict(
        type='PartConvKernelHead',
        num_classes=19,
        num_thing_classes=8,
        num_stuff_classes=11,
        num_part_classes=23,
        cat_stuff_mask=True,
        conv_kernel_size=conv_kernel_size,
        feat_downsample_stride=2,
        feat_refine_stride=1,
        feat_refine=False,
        use_binary=True,
        num_inst_convs=1,
        num_seg_convs=1,
        num_part_cnovs=1,
        conv_normal_init=True,
        localization_fpn=dict(
            type='SemanticFPNWrapper',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            num_aux_convs=2,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        num_proposals=num_proposals,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        num_cls_fcs=1,
        in_channels=256,
        feat_transform_cfg=None,
        loss_rank=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
        loss_seg=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_part=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=4.0)),
    # iterative kernel
    roi_head=dict(
        type='PartKernelIterHead',
        eval_spec_path='./panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml',
        output_dir='./work_dirs/part_res/',
        cityscapes=True,
        num_stages=num_stages,
        num_thing_classes=8,
        num_stuff_classes=11,
        num_part_classes=23,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
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
                loss_mask=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(
                    type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)) for _ in range(num_stages)
        ]),
)


load_from = "/mnt/lustre/lixiangtai/pretrained/knet/knet_r50_city.pth"
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
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

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8)  # actual epoch = 8 * 8 = 64