# dataset settings
dataset_type = 'CityscapesPanopticPartDataset'
data_root = 'data/cityscapes/'

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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
    val=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/instancesonly_filtered_gtFine_val.json',
            panoptic_ann=data_root + "annotations/cityscapes_panoptic_val.json"
        ),
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val',
        part_prefix=data_root + 'gtFinePanopticParts/val/',
        panoptic_part_eval_config=dict(
            eval_spec_path="panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml",
            basepath_gt="data/cityscapes/gtFinePanopticParts/val",
            basepath_pred="work_dirs/part_res",
            images_json="data/cityscapes/annotations/val_images.json",
            save_dir="work_dirs/part_res/save"
        ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=dict(
            ins_ann=data_root + 'annotations/instancesonly_filtered_gtFine_val.json',
            panoptic_ann=data_root + "annotations/cityscapes_panoptic_val.json"
            # ins_ann=data_root + 'annotations/instance_mini_val.json',
            # panoptic_ann=data_root + 'annotations/panoptic_mini_val.json'
        ),
        img_prefix=data_root + 'leftImg8bit/val/',
        seg_prefix=data_root + 'gtFine/cityscapes_panoptic_val',
        part_prefix=data_root + 'gtFinePanopticParts/val/',
        panoptic_part_eval_config=dict(
            eval_spec_path="panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml",
            basepath_gt="data/cityscapes/gtFinePanopticParts/val",
            basepath_pred="work_dirs/part_res",
            images_json="data/cityscapes/annotations/val_images.json",
            # images_json="data/cityscapes/annotations/mini_val_images.json",
            save_dir="work_dirs/part_res/save"
        ),
        pipeline=test_pipeline))
