num_stages = 3
conv_kernel_size = 1
_base_ = [
    './pascal_part_r50_baseline.py',
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerDIY',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    rpn_head=dict(
        localization_fam=dict(
            type='UperNetAlignHead',
            in_channels=[256, 256, 256, 256],
            out_channels=256),
        localization_fpn=dict(
            num_aux_convs=1,
        ),
    ),
    # iterative kernel
    roi_head=dict(
        mask_head=[
            dict(
                thing_part_stuff_update=False,
                thing_stuff_part_update=False,
                thing_part_stuff_attention=False,
                thing_stuff_part_attention=False,

                type='PartKernelUpdateHead',
                num_classes=116,
                num_thing_classes=20,
                num_stuff_classes=39,
                num_part_classes=57,
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

