import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead
from panopticapi.utils import rgb2id
import panoptic_parts as pp
from models.assigner.mask_pseudo_sampler import MaskPseudoSampler
from panoptic_parts.specs.eval_spec import PartPQEvalSpec
import json
import numpy as np
from PIL import Image
import os


@HEADS.register_module()
class PartKernelIterHead(BaseRoIHead):

    def __init__(self,
                 merge_gt_part=False,
                 merge_gt_panoptic=False,
                 num_stages=6,
                 recursive=False,
                 assign_stages=5,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 merge_cls_scores=False,
                 do_panoptic=False,
                 post_assign=False,
                 hard_target=False,
                 merge_joint=False,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 mask_head=dict(
                     type='KernelUpdateHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_out_stride=4,
                 train_cfg=None,
                 test_cfg=None,
                 eval_spec_path=None,
                 images_json=None,
                 num_part_classes=23,
                 output_dir=None,
                 cityscapes=False,
                 **kwargs):
        self.merge_gt_part = merge_gt_part
        self.merge_gt_panoptic = merge_gt_panoptic
        self.evalspec = pp.specs.eval_spec.PartPQEvalSpec(
            './panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml')
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.cityscapes = cityscapes
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.images_json = images_json
        self.eval_spec_path = eval_spec_path
        self.eval_spec = PartPQEvalSpec(self.eval_spec_path)
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.merge_cls_scores = merge_cls_scores
        self.recursive = recursive
        self.post_assign = post_assign
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.merge_joint = merge_joint
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_part_classes = num_part_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.num_proposals = num_proposals
        self.ignore_label = ignore_label
        self.void = self.eval_spec.ignore_label
        self.sid_pid2part_seg_label = self.eval_spec.eval_sid_pid2eval_pid_flat
        self.sids2part_seg_ids, self.part_seg_ids2eval_pids_per_sid = self._prepare_mappings(
            self.sid_pid2part_seg_label, self.void)
        self.seg2stuff_ids = {i: sid for i, sid in enumerate(sorted(self.eval_spec.eval_sid_stuff))}
        self.ins2thing_ids = {i: tid for i, tid in enumerate(sorted(self.eval_spec.eval_sid_things))}
        super(PartKernelIterHead, self).__init__(
            mask_head=mask_head, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(
                    self.mask_sampler[stage], MaskPseudoSampler), \
                    'Sparse Mask only support `MaskPseudoSampler`'

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if self.recursive:
            for i in range(self.num_stages):
                self.mask_head[i] = self.mask_head[0]

    def _mask_forward(self, stage, x, object_feats, mask_preds, img_metas):
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats = mask_head(
            x, object_feats, mask_preds, img_metas=img_metas)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds
        mask_results = dict(
            cls_score=cls_score,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)

        return mask_results

    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_bboxes=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None,
                      gt_part=None,
                      gt_part_cls=None):

        num_imgs = len(img_metas)
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()

        if cls_score is not None:
            prev_cls_score = cls_score.detach()
        else:
            prev_cls_score = [None] * num_imgs

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        assign_results = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']

            if self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

            sampling_results = []
            if stage < self.assign_stages:
                assign_results = []
            for i in range(num_imgs):
                if stage < self.assign_stages:
                    mask_for_assign = prev_mask_preds[i][:self.num_proposals]
                    if prev_cls_score[i] is not None:
                        cls_for_assign = prev_cls_score[
                                             i][:self.num_proposals, :self.num_thing_classes]
                    else:
                        cls_for_assign = None
                    assign_result = self.mask_assigner[stage].assign(
                        mask_for_assign, cls_for_assign, gt_masks[i],
                        gt_labels[i], img_metas[i])
                    assign_results.append(assign_result)
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], scaled_mask_preds[i], gt_masks[i])
                sampling_results.append(sampling_result)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                gt_masks,
                gt_labels,
                self.train_cfg[stage],
                True,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls,
                gt_part=gt_part, gt_part_cls=gt_part_cls
            )

            single_stage_loss = self.mask_head[stage].loss(
                object_feats,
                cls_score,
                scaled_mask_preds,
                *mask_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f's{stage}_{key}'] = value * self.stage_loss_weights[stage]

            if not self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    imgs_whwh=None,
                    rescale=False):

        # Decode initial proposals
        num_imgs = len(img_metas)
        # num_proposals = proposal_feats.size(1)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id])
                single_result = self.merge_to_pps(cls_score[img_id], scaled_mask_preds[img_id], self.test_cfg,
                                                  img_metas[img_id], single_result)

                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                    self.test_cfg.max_per_img, sorted=True)
                mask_indices = topk_indices // num_classes
                labels_per_img = topk_indices % num_classes
                masks_per_img = scaled_mask_preds[img_id][mask_indices]
                single_result = self.mask_head[-1].get_seg_masks(
                    masks_per_img, labels_per_img, scores_per_img,
                    self.test_cfg, img_metas[img_id])
                results.append(single_result)
        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('SparseMask does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        C, H, W = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, C, -1)).view(
            num_imgs, num_proposals, H, W)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

    def get_panoptic(self, cls_scores, mask_preds, test_cfg, img_meta):
        # resize mask predictions back
        thing_scores = cls_scores[:self.num_proposals][:, :self.num_thing_classes]

        thing_mask_preds = mask_preds[:self.num_proposals]
        thing_scores, topk_indices = thing_scores.flatten(0, 1).topk(self.test_cfg.max_per_img, sorted=True)

        mask_indices = topk_indices // self.num_thing_classes
        thing_labels = topk_indices % self.num_thing_classes
        masks_per_img = thing_mask_preds[mask_indices]
        thing_masks = self.mask_head[-1].rescale_masks(masks_per_img, img_meta)
        if not self.merge_joint:
            thing_masks = thing_masks > test_cfg.mask_thr
        thing_masks = thing_masks > test_cfg.mask_thr
        bbox_result, segm_result = self.mask_head[-1].segm2result(
            thing_masks, thing_labels, thing_scores)

        stuff_scores = cls_scores[self.num_proposals:self.num_proposals + self.num_stuff_classes,
                       self.num_thing_classes:self.num_thing_classes + self.num_stuff_classes].diag()

        stuff_scores, stuff_inds = torch.sort(stuff_scores, descending=True)
        stuff_masks = mask_preds[self.num_proposals:self.num_proposals + self.num_stuff_classes][stuff_inds]
        stuff_masks = self.mask_head[-1].rescale_masks(stuff_masks, img_meta)
        if not self.merge_joint:
            stuff_masks = stuff_masks > test_cfg.mask_thr

        if self.merge_joint:
            stuff_labels = stuff_inds + self.num_thing_classes
            panoptic_result = self.merge_stuff_thing_stuff_joint(thing_masks, thing_labels,
                                                                 thing_scores, stuff_masks,
                                                                 stuff_labels, stuff_scores,
                                                                 test_cfg.merge_stuff_thing)
        else:
            stuff_labels = stuff_inds
            panoptic_result = self.merge_stuff_thing(thing_masks, thing_labels,
                                                     thing_scores, stuff_masks,
                                                     stuff_labels, stuff_scores,
                                                     test_cfg.merge_stuff_thing)
        return bbox_result, segm_result, panoptic_result

    def split_thing_stuff(self, mask_preds, det_labels, cls_scores):
        thing_scores = cls_scores[:self.num_proposals]
        thing_masks = mask_preds[:self.num_proposals]
        thing_labels = det_labels[:self.num_proposals]

        stuff_labels = det_labels[self.num_proposals:]
        stuff_labels = stuff_labels - self.num_thing_classes + 1
        stuff_masks = mask_preds[self.num_proposals:]
        stuff_scores = cls_scores[self.num_proposals:]

        results = (thing_masks, thing_labels, thing_scores, stuff_masks,
                   stuff_labels, stuff_scores)
        return results

    def merge_stuff_thing(self,
                          thing_masks,
                          thing_labels,
                          thing_scores,
                          stuff_masks,
                          stuff_labels,
                          stuff_scores,
                          merge_cfg=None):

        H, W = thing_masks.shape[-2:]
        panoptic_seg = thing_masks.new_zeros((H, W), dtype=torch.int32)
        thing_masks = thing_masks.to(
            dtype=torch.bool, device=panoptic_seg.device)
        stuff_masks = stuff_masks.to(
            dtype=torch.bool, device=panoptic_seg.device)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-thing_scores)
        current_segment_id = 0
        segments_info = []
        # Add instances one-by-one, check for overlaps with existing ones
        for inst_id in sorted_inds:
            score = thing_scores[inst_id].item()
            if score < merge_cfg.instance_score_thr:
                break
            mask = thing_masks[inst_id]  # H,W
            mask_area = mask.sum().item()

            if mask_area == 0:
                continue

            intersect = (mask > 0) & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()

            if intersect_area * 1.0 / mask_area > merge_cfg.iou_thr:
                continue

            if intersect_area > 0:
                mask = mask & (panoptic_seg == 0)

            mask_area = mask.sum().item()
            if mask_area == 0:
                continue

            current_segment_id += 1
            panoptic_seg[mask.bool()] = current_segment_id
            segments_info.append({
                'id': current_segment_id,
                'isthing': True,
                'score': score,
                'category_id': thing_labels[inst_id].item(),
                'instance_id': inst_id.item(),
            })

        # Add semantic results to remaining empty areas
        sorted_inds = torch.argsort(-stuff_scores)
        sorted_stuff_labels = stuff_labels[sorted_inds]
        # paste semantic masks following the order of scores
        processed_label = []
        for semantic_label in sorted_stuff_labels:
            semantic_label = semantic_label.item()
            if semantic_label in processed_label:
                continue
            processed_label.append(semantic_label)
            sem_inds = stuff_labels == semantic_label
            sem_masks = stuff_masks[sem_inds].sum(0).bool()
            mask = sem_masks & (panoptic_seg == 0)
            mask_area = mask.sum().item()
            if mask_area < merge_cfg.stuff_max_area:
                continue

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append({
                'id': current_segment_id,
                'isthing': False,
                'category_id': semantic_label,
                'area': mask_area,
            })
        return panoptic_seg.cpu().numpy(), segments_info

    def merge_stuff_thing_stuff_joint(self,
                                      thing_masks,
                                      thing_labels,
                                      thing_scores,
                                      stuff_masks,
                                      stuff_labels,
                                      stuff_scores,
                                      merge_cfg=None):

        H, W = thing_masks.shape[-2:]
        panoptic_seg = thing_masks.new_zeros((H, W), dtype=torch.int32)

        total_masks = torch.cat([thing_masks, stuff_masks], dim=0)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        segments_info = []
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg.overlap_thr:
                    continue
                current_segment_id += 1

                panoptic_seg[mask] = current_segment_id

                if isthing:
                    segments_info.append({
                        'id': current_segment_id,
                        'isthing': isthing,
                        'score': total_scores[k].item(),
                        'category_id': pred_class,
                        'instance_id': k.item(),
                    })
                else:
                    segments_info.append({
                        'id': current_segment_id,
                        'isthing': isthing,
                        'category_id': pred_class - self.num_thing_classes,
                        'area': mask_area,
                    })

        return panoptic_seg.cpu().numpy(), segments_info

    def _prepare_mappings(self, sid_pid2part_seg_label, void):
        import numpy as np
        # Get the maximum amount of part_seg labels
        num_part_seg_labels = np.max(
            list(sid_pid2part_seg_label.values()))

        sids2part_seg_ids = dict()
        for class_key in sid_pid2part_seg_label.keys():
            class_id = class_key // 100
            if class_id in sids2part_seg_ids.keys():
                if sid_pid2part_seg_label[class_key] not in sids2part_seg_ids[class_id]:
                    sids2part_seg_ids[class_id].append(sid_pid2part_seg_label[class_key])
                else:
                    raise ValueError(
                        'A part seg id can only be shared between different semantic classes, not within a single semantic class.')
            else:
                sids2part_seg_ids[class_id] = [sid_pid2part_seg_label[class_key]]

        sids2pids_eval = dict()
        for class_key in sid_pid2part_seg_label.keys():
            class_id = class_key // 100
            if class_id in sids2pids_eval.keys():
                if class_key % 100 not in sids2pids_eval[class_id]:
                    sids2pids_eval[class_id].append(class_key % 100)
            else:
                sids2pids_eval[class_id] = [class_key % 100]

        part_seg_ids2eval_pids_per_sid = dict()
        for class_key in sids2part_seg_ids.keys():
            tmp = np.ones(num_part_seg_labels + 1, np.uint8) * void
            tmp[sids2part_seg_ids[class_key]] = sids2pids_eval[class_key]
            part_seg_ids2eval_pids_per_sid[class_key] = tmp

        return sids2part_seg_ids, part_seg_ids2eval_pids_per_sid

    def merge_to_pps(self, cls_scores, mask_preds, test_cfg, img_meta, single_result):

        pred_pan_flat = single_result[2][0]
        segments = single_result[2][1]
        part_scores = cls_scores[-self.num_part_classes:, -self.num_part_classes:].diag()
        _, part_inds = torch.sort(part_scores, descending=False)
        part_masks = mask_preds[-self.num_part_classes:, :, :]
        part_masks = self.mask_head[-1].rescale_masks(part_masks, img_meta)
        part_masks = (part_masks > test_cfg.mask_thr).cpu().numpy()
        h, w = img_meta['ori_shape'][:2]
        if self.merge_gt_panoptic and not self.training:
            thing_ids2ins = {v: k for k, v in self.ins2thing_ids.items()}
            stuff_ids2seg = {v: k for k, v in self.seg2stuff_ids.items()}
            file_name = img_meta['ori_filename'].split('/')[-1].replace('leftImg8bit.png', 'gtFine_panoptic.png')
            x = np.array(Image.open(os.path.join('data/cityscapes/gtFine/cityscapes_panoptic_val', file_name)))
            annotations = json.load(open('data/cityscapes/annotations/cityscapes_panoptic_val.json'))
            x = rgb2id(x)
            segms = [ann for ann in annotations['annotations'] if ann['file_name'] == file_name][0]['segments_info']
            segms = [s for s in segms if s['category_id'] in self.ins2thing_ids.values() or s[
                'category_id'] in self.seg2stuff_ids.values()]
            for s in segms:
                if s['category_id'] in self.ins2thing_ids.values():
                    s['isthing'] = True
                    s['score'] = 1
                    s['instance_id'] = s['id']
                    s['category_id'] = thing_ids2ins[s['category_id']]
                else:
                    s['isthing'] = False
                    s['category_id'] = stuff_ids2seg[s['category_id']]
            pred_pan_flat = x
            segments = segms
        if self.merge_gt_part and not self.training:
            uids = np.array(Image.open(os.path.join('data/cityscapes/gtFinePanopticParts/val',
                                                    img_meta['ori_filename'].replace('leftImg8bit.png',
                                                                                     'gtFinePanopticParts.tif'))))

            sids, _, pids, sid_pids = pp.decode_uids(uids, return_sids_pids=True)
            sid_pid2part_id = {k: v - 1 for k, v in self.evalspec.eval_sid_pid2eval_pid_flat.items()}
            gt_part = np.ones_like(sids, dtype=np.long) * 23
            for sid_pid in sid_pid2part_id.keys():
                mask = sid_pids == sid_pid
                gt_part[mask] = sid_pid2part_id[sid_pid]
            gt_part = torch.from_numpy(gt_part)
            gt_part = F.one_hot(gt_part).permute(2, 0, 1)[:23, :, :]
            part_masks = gt_part.numpy()
        pred_part = np.zeros((h, w), dtype=np.long)
        for inds in part_inds:
            pred_part = np.where(part_masks[inds.item()] != 0, inds.item() + 1, pred_part)
            # pred_part[part_masks[inds.item()]] = inds.item() + 1  # [1,23]

        segment_count_per_cat = dict()
        class_canvas = np.ones((h, w), dtype=np.int32) * self.void
        inst_canvas = np.zeros((h, w), dtype=np.int32)
        # TODO(daan): check whether we can also set part_canvas init to 255
        part_canvas = np.zeros((h, w), dtype=np.int32)
        for segment in segments:
            segment_id = segment['id']
            cat_id = segment['category_id']
            if segment['isthing']:
                cat_id = self.ins2thing_ids[cat_id]
            else:
                cat_id = self.seg2stuff_ids[cat_id]

            # Increase the segment count per category
            if cat_id in segment_count_per_cat.keys():
                segment_count_per_cat[cat_id] += 1
            else:
                segment_count_per_cat[cat_id] = 1

            if segment_count_per_cat[cat_id] > 255:
                raise ValueError(
                    'More than 255 instances for category_id > {}. This is currently not yet supported.'.format(cat_id))

            mask = pred_pan_flat == segment_id

            # Loop over all scene-level categories
            if cat_id in self.eval_spec.eval_sid_parts:
                # If category has parts
                # Check what pids are possible for the sid
                plausible_parts = self.sids2part_seg_ids[cat_id]
                plausible_parts_mask = np.isin(pred_part, plausible_parts)

                # Get the mapping from part_seg ids to evaluation pids, given the sid
                part_seg_ids2eval_pids = self.part_seg_ids2eval_pids_per_sid[cat_id]
                part_canvas[mask] = self.void

                # Convert the part seg ids to the desired evaluation pids, and store them in the tensor with part labels
                part_canvas[np.logical_and(mask, plausible_parts_mask)] = part_seg_ids2eval_pids[
                    pred_part[np.logical_and(mask, plausible_parts_mask)]]

                # Store the category id and instance id in the respective tensors
                class_canvas[mask] = cat_id
                inst_canvas[mask] = segment_count_per_cat[cat_id]
            else:
                mask = pred_pan_flat == segment_id

                # Store the category id and instance id in the respective tensors
                class_canvas[mask] = cat_id
                inst_canvas[mask] = segment_count_per_cat[cat_id]
                # Store a dummy part id
                part_canvas[mask] = 1

        pred_pan_part = np.stack([class_canvas, inst_canvas, part_canvas], axis=2)
        img_pan_part = Image.fromarray(pred_pan_part.astype(np.uint8))
        filename = img_meta['ori_filename']
        if self.cityscapes:
            filename = filename.split('/')[-1].replace('leftImg8bit', 'gtFine_panoptic')
        else:
            filename = filename.replace('jpg', 'png')
        img_pan_part.save(os.path.join(self.output_dir, filename))
        return single_result
