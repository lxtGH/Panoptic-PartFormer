import torch
import io

from panopticapi.utils import id2rgb
from PIL import Image


def part_inst2inst_mask(gt_part):
    gt_part_seg = torch.zeros_like(gt_part[0])
    for i in range(gt_part.shape[0]):
        gt_part_seg = torch.where(gt_part[i] != 0, gt_part[i], gt_part_seg)
    classes = gt_part.unique()
    ins_masks = []
    ins_labels = []
    for i in classes:
        ins_labels.append(i)
        ins_masks.append(gt_part_seg == i)
    ins_labels = torch.stack(ins_labels)
    ins_masks = torch.stack(ins_masks)
    return ins_labels.long(), ins_masks.float()


def part2ins_masks(gt_part, ignore_label=255, label_shift=19):
    classes = torch.unique(gt_part)
    ins_masks = []
    ins_labels = []
    for i in classes:
        if i in ignore_label:
            continue
        ins_labels.append(i)
        ins_masks.append(gt_part == i)
    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_part.new_zeros(size=[0])
        ins_masks = gt_part.new_zeros(size=[0, gt_part.shape[-2], gt_part.shape[-1]])
    return ins_labels.long(), ins_masks.float()


def sem2ins_masks(gt_sem_seg, ignore_label=255, label_shift=79, thing_label_in_seg=0):
    classes = torch.unique(gt_sem_seg)
    ins_masks = []
    ins_labels = []
    for i in classes:
        # skip ignore class 255 and "special thing class" in semantic seg
        # if i == ignore_label or i in thing_label_in_seg:
        if i == ignore_label:
            continue
        ins_labels.append(i)
        ins_masks.append(gt_sem_seg == i)
    # 0 is the special thing class in semantic seg, so we also shift it by 1
    # Thus, 0-79 is foreground classes of things (similar in instance seg)
    # 80-151 is foreground classes of stuffs (shifted by the original index)
    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_sem_seg.new_zeros(size=[0])
        ins_masks = gt_sem_seg.new_zeros(size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return ins_labels.long(), ins_masks.float()


def sem2ins_masks_cityscapes(gt_sem_seg,
                             ignore_label=255,
                             label_shift=8,
                             thing_label_in_seg=list(range(11, 19))):
    """
        Shift the cityscapes semantic labels to instance labels and masks.
    """
    # assert label range from 0-18 (255)
    classes = torch.unique(gt_sem_seg)
    ins_masks = []
    ins_labels = []
    for i in classes:
        # skip ignore class 255 and "special thing class" in semantic seg
        if i == ignore_label or i in thing_label_in_seg:  # train_id
            continue
        ins_labels.append(i)
        ins_masks.append(gt_sem_seg == i)
    # For cityscapes, 0-7 is foreground classes of things (similar in instance seg)
    # 8-18 is foreground classes of stuffs (shifted by the original index)

    if len(ins_labels) > 0:
        ins_labels = torch.stack(ins_labels) + label_shift
        ins_masks = torch.cat(ins_masks)
    else:
        ins_labels = gt_sem_seg.new_zeros(size=[0])
        ins_masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return ins_labels.long(), ins_masks.float()


def encode_panoptic(panoptic_results):
    panoptic_img, segments_info = panoptic_results
    with io.BytesIO() as out:
        Image.fromarray(id2rgb(panoptic_img)).save(out, format='PNG')
        return out.getvalue(), segments_info
