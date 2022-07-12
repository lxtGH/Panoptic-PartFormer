# code from the original K-Net.
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER


@TRANSFORMER_LAYER.register_module()
class KernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=3,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features


@TRANSFORMER_LAYER.register_module()
class TreeLikeKernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=3,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 num_things=100,
                 num_stuff=11,
                 num_parts=23,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')
                 ):
        super(TreeLikeKernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.num_things = num_things
        self.num_stuff = num_stuff
        self.num_parts = num_parts

    def forward(self, update_feature, input_feature):
        assert self.num_things + self.num_stuff + self.num_parts == update_feature.size(1)

        N = update_feature.size(0)
        things_stuff_update_feature = update_feature[:, :self.num_things+self.num_stuff]
        things_stuff_input_feature = input_feature[:, :self.num_things+self.num_stuff]

        things_stuff_update_feature = self.update(things_stuff_update_feature, things_stuff_input_feature)

        things_stuff_update_feature = things_stuff_update_feature.reshape(N, self.num_things + self.num_stuff, -1)

        things_update_feature = things_stuff_update_feature[:, :self.num_things]
        part_update_feature = update_feature[:, self.num_things+self.num_stuff:]
        part_input_feature = input_feature[:, self.num_things+self.num_stuff:]

        things_part_update_feature = torch.cat([things_update_feature, part_update_feature], dim=1)
        things_part_input_feature = torch.cat([things_update_feature.unsqueeze(2), part_input_feature], dim=1)
        # print(things_part_update_feature.shape)
        # print(things_part_input_feature.shape)
        things_part_update_feature = self.update(things_part_update_feature, things_part_input_feature)
        things_part_update_feature = things_part_update_feature.reshape(N, self.num_things + self.num_parts, -1)

        things_features = things_part_update_feature[:, :self.num_things]
        stuff_features = things_stuff_update_feature[:, self.num_things:]
        part_features = things_part_update_feature[:, self.num_things:]
        final_feature = torch.cat([things_features, stuff_features, part_features], dim=1)
        final_feature = final_feature.permute(1, 0, 2)

        return final_feature

    def update(self, update_feature, input_feature):
        update_feature = update_feature.reshape(-1, self.in_channels)
        num_proposals = update_feature.size(0)
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)

        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features