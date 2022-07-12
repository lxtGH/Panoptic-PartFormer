"""
    Unified Panoptic Part Network For End-to-End Panoptic Part Segmentation.
"""
import torch
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

from models.utils.utils import sem2ins_masks, sem2ins_masks_cityscapes, part2ins_masks


@DETECTORS.register_module()
class PartNet(TwoStageDetector):

    def __init__(self,
                 *args,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 num_part_classes=0,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cityscapes=False,
                 with_boundary=False,
                 **kwargs):
        super(PartNet, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'KNet does not support external proposals'
        self.with_boundary = with_boundary
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_part_classes = num_part_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.ignore_label = ignore_label
        self.cityscapes = cityscapes  # whether to train the cityscape panoptic segmentation

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      gt_part=None,
                      **kwargs):
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        super(TwoStageDetector, self).forward_train(img, img_metas)
        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        gt_part_seg = []
        gt_part_cls = []
        gt_boundary = []
        laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], device=gt_part.device).reshape(1, 1, 3,
                                                                                                            3).float()
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask.to_tensor(torch.float, gt_labels[0].device)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)

            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by zero when forming a batch
                # need to convert them from 0 to ignore
                gt_semantic_seg[i, :, img_metas[i]['img_shape'][0]:, :] = self.ignore_label
                gt_semantic_seg[i, :, :, img_metas[i]['img_shape'][1]:] = self.ignore_label

                if self.cityscapes:
                    sem_labels, sem_seg = sem2ins_masks_cityscapes(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes)
                else:
                    sem_labels, sem_seg = sem2ins_masks(
                        gt_semantic_seg[i],
                        ignore_label=self.ignore_label,
                        label_shift=self.num_thing_classes,
                        thing_label_in_seg=self.thing_label_in_seg)

                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(mask_tensor.new_zeros((1, assign_H, assign_W)))
                    gt_sem_cls.append(
                        torch.tensor([self.num_stuff_classes + self.num_thing_classes], device=mask_tensor.device))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                    gt_sem_cls.append(sem_labels)
            else:
                gt_sem_seg = None
                gt_sem_cls = None

            if gt_part is not None:
                gt_part[i, :, :, img_metas[i]['img_shape'][1]:] = self.ignore_label
                gt_part[i, :, img_metas[i]['img_shape'][0]:, :] = self.ignore_label
                part_labels, part_seg = part2ins_masks(gt_part[i], ignore_label=[self.ignore_label],
                                                       label_shift=self.num_thing_classes + self.num_stuff_classes)
                if part_seg.shape[0] == 0:
                    gt_part_seg.append(
                        mask_tensor.new_zeros(
                            (1, assign_H, assign_W)))
                    gt_part_cls.append(
                        torch.tensor([self.num_part_classes + self.num_stuff_classes + self.num_thing_classes],
                                     device=mask_tensor.device))
                    gt_boundary.append(mask_tensor.new_zeros(gt_part.size()[2:]))
                else:
                    gt_part_seg.append(
                        F.interpolate(part_seg[None], (assign_H, assign_W), mode='bilinear', align_corners=False)[0])
                    # part_boundary = torch.zeros((assign_H, assign_W), device=part_seg.device)
                    part_boundary = part_seg.new_zeros(part_seg.size()[1:])
                    for j in range(part_labels.numel()):
                        b = F.conv2d(part_seg[j][None][None], laplacian_kernel, padding=1).squeeze()
                        b[b <= 0] = 0
                        b[b > 0] = 1
                        part_boundary = torch.where(b != 0, b, part_boundary)
                    gt_boundary.append(part_boundary)
                    gt_part_cls.append(part_labels)
            else:
                gt_part_cls = None
                gt_part_seg = None

            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])

        gt_masks = gt_masks_tensor
        x = self.extract_feat(img)

        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,
                                                  gt_labels, gt_sem_seg,
                                                  gt_sem_cls, gt_part_seg, gt_part_cls, gt_boundary=gt_boundary)
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results

        losses = self.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            gt_masks,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_bboxes=gt_bboxes,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            gt_part=gt_part_seg,
            gt_part_cls=gt_part_cls,
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds, part_preds) = rpn_results
        segm_results = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)
        return segm_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats,
                                               dummy_img_metas)
        return roi_outs
