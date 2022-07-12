from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack


class PSPModule(BaseModule):
    def __init__(self, in_features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


@NECKS.register_module()
class UperNetAlignHead(BaseModule):

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256,
                 conv3x3_type="conv"):
        super(UperNetAlignHead, self).__init__()

        # self.ppm = PSPModule(in_channels[-1], out_features=out_channels)
        self.dcn = DeformConv2dPack(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        self.fpn_in = []
        for fpn_inplane in in_channels[:-1]:
            self.fpn_in.append(
                ConvModule(fpn_inplane, out_channels, kernel_size=1, norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='ReLU'),
                           inplace=False)
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(in_channels) - 1):
            self.fpn_out.append(
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                           norm_cfg=dict(type='BN2d')))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=out_channels, outplane=out_channels // 2)
                )

            self.fpn_out = nn.ModuleList(self.fpn_out)
            self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

    def forward(self, conv_out):
        f = conv_out[-1]
        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))

        output_size = conv_out[1].size()[2:]
        fusion_list = []

        for i in range(0, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        x = fusion_list[0]
        for i in range(1, len(fusion_list)):
            x += fusion_list[i]

        return self.dcn(x)
