import os.path as osp
import numpy as np
from PIL import Image
import mmcv
import os
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.core import BitmapMasks, PolygonMasks
import panoptic_parts as pp
import collections




@PIPELINES.register_module()
class LoadPartAnnotations(LoadAnnotations):
    def __init__(self, with_part=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_part = with_part

    def _load_part(self, results):
        sid_pid = [2401, 2402, 2403, 2404,
                   2501, 2502, 2503, 2504,
                   2601, 2602, 2603, 2604, 2605,
                   2701, 2702, 2703, 2704, 2705,
                   2801, 2802, 2803, 2804, 2805]
        import panoptic_parts as pp

        filename = osp.join(results['part_prefix'], results['ann_info']['part_map'])
        img = np.array(Image.open(filename))
        sids, iids, pids = pp.decode_uids(img)
        pids_true = pids != -1
        gt_part = np.where(pids_true, 100 * sids + pids, 255)
        for i, x in enumerate(sid_pid):
            gt_part = np.where(gt_part == x, i, gt_part)  # 2401--0, 2402--1 ... 2805--22
        results['gt_part'] = gt_part
        results['part_fields'].append('gt_part')
        return results

    def __call__(self, results):
        results = super(LoadPartAnnotations, self).__call__(results)
        if self.with_part:
            results = self._load_part(results)
        return results


@PIPELINES.register_module()
class LoadPascalPartAnnotation:
    def __init__(self, data_spec_path, eval_spec_path, **kwargs):
        self.dataspec = pp.specs.dataset_spec.DatasetSpec(data_spec_path)
        self.evalspec = pp.specs.eval_spec.PartPQEvalSpec(eval_spec_path)
        self.sid_pid = []
        for k in self.evalspec.eval_sid_pid2eval_pid_flat.keys():
            self.sid_pid.append(k)
        self.sid_pid = sorted(self.sid_pid)

    def _num_instances_per_sid(self, uids):
        uids_unique = np.unique(np.array(uids, dtype=np.int32))
        _, _, _, sids_iids = pp.decode_uids(uids_unique, return_sids_iids=True)
        sids_iids_unique = np.unique(sids_iids)
        sid2Ninstances = collections.defaultdict(lambda: 0)
        for sid_iid in sids_iids_unique:
            sid, iid, _ = pp.decode_uids(sid_iid)
            if iid >= 0:
                sid2Ninstances[sid] += 1
        return sid2Ninstances

    def __call__(self, results):
        img_info = results['img_info']
        img = np.array(Image.open(os.path.join(results['part_prefix'], img_info['part_file'])))
        sids, iids, pids, sid_iids = pp.decode_uids(img, return_sids_iids=True, experimental_dataset_spec=self.dataspec)
        multi_inst = self._num_instances_per_sid(sid_iids)
        gt_masks = []
        gt_semantic_seg = np.ones_like(img) * 255
        gt_labels = []
        stuff_id = sorted(self.evalspec.eval_sid_stuff)
        for class_id in np.unique(sids):
            if class_id == 0:
                continue
            if class_id in self.evalspec.eval_sid_things:
                mask = sids == class_id
                if class_id in multi_inst:
                    inst_mask = np.zeros_like(iids)
                    inst_mask[mask] = iids[mask]
                    for inst_id in np.unique(inst_mask):
                        if inst_id == -1:
                            continue
                        imask = (inst_mask == inst_id) & mask
                        gt_masks.append(imask.astype(np.uint8))
                        gt_labels.append(class_id - 1)
                else:
                    gt_masks.append(mask.astype(np.uint8))
                    gt_labels.append(class_id - 1)
            if class_id in stuff_id:  # stuff
                gt_semantic_seg = np.where(sids == class_id, stuff_id.index(class_id), gt_semantic_seg)
        pids_true = pids != -1
        gt_part = np.where(pids_true, 100 * sids + pids, 255)
        for i, x in enumerate(self.sid_pid):
            gt_part = np.where(gt_part == x, i, gt_part)  # 2401--0, 2402--1 ... 2805--22

        results['gt_part'] = gt_part
        results['part_fields'].append('gt_part')

        results['gt_labels'] = gt_labels  ##thing label

        gt_masks = BitmapMasks(gt_masks, img.shape[0], img.shape[1])
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')

        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

