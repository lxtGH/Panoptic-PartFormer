# dataset settings
dataset_type = 'PascalPanopticPartDataset'
data_root = 'data/VOCdevkit/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPascalPartAnnotation', data_spec_path='./panoptic_parts/specs/dataset_specs/ppp_datasetspec.yaml',
         eval_spec_path='./panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_baseline_evalspec.yaml', ),
    dict(
        type='PartResize', img_scale=(1333, 800), multiscale_mode='range', keep_ratio=True),
    dict(type='PartRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PartPad', size_divisor=32),
    dict(type='PartDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_part']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            img_prefix=data_root + 'VOC2010/JPEGImages/',
            part_prefix=data_root + 'labels_57part/training/',
            panoptic_part_eval_config=dict(
                eval_spec_path="panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml",
                basepath_gt="data/VOCdevkit/labels_57part/validation",
                basepath_pred="work_dirs/pascal_part_res",
                images_json="data/VOCdevkit/labels_57part/val_images.json",
                save_dir="work_dirs/pascal_part_res/save"
            ),
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'VOC2010/JPEGImages/',
        seg_prefix=data_root + 'gt_panoptic',
        panoptic_gt_json=data_root + 'panoptic.json',
        part_prefix=data_root + 'labels_57part/validation/',
        panoptic_part_eval_config=dict(
            eval_spec_path="panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml",
            basepath_gt="data/VOCdevkit/labels_57part/validation/",
            basepath_pred="work_dirs/pascal_part_res",
            images_json="data/VOCdevkit/labels_57part/val_images.json",
            save_dir="work_dirs/pascal_part_res/save"
        ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'VOC2010/JPEGImages/',
        seg_prefix=data_root + 'gt_panoptic',
        # panoptic_gt_json=data_root + 'mini_panoptic.json',
        # part_prefix=data_root + 'labels_57part/mini_val/',
        panoptic_gt_json=data_root + 'panoptic.json',
        part_prefix=data_root + 'labels_57part/validation/',
        panoptic_part_eval_config=dict(
            eval_spec_path="panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml",
            # basepath_gt="data/VOCdevkit/labels_57part/mini_val/",
            basepath_gt="data/VOCdevkit/labels_57part/validation/",
            basepath_pred="work_dirs/pascal_part_res",
            images_json="data/VOCdevkit/labels_57part/val_images.json",
            # images_json="data/VOCdevkit/labels_57part/mini_val_images.json",
            save_dir="work_dirs/pascal_part_res/save"
        ),
        pipeline=test_pipeline))
