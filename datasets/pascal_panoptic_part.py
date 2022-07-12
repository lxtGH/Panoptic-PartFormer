from mmdet.datasets.pipelines import Compose

import contextlib
import io
import os
import glob
import tempfile
import logging
import os.path as osp
from collections import OrderedDict

import pycocotools.mask as maskUtils

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from datasets.coco_panoptic import parse_pq_results, _print_panoptic_results
from mmdet.datasets import CustomDataset
from panoptic_parts.specs.eval_spec import PartPQEvalSpec


@DATASETS.register_module()
class PascalPanopticPartDataset:
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def __init__(self, pipeline=None, img_prefix=None, part_prefix=None, test_mode=False,
                 panoptic_part_eval_config=None, panoptic_gt_json=None, seg_prefix=None, **kwargs):
        self.seg_prefix = seg_prefix
        self.panoptic_gt_json = panoptic_gt_json
        self.panoptic_part_eval_config = panoptic_part_eval_config
        self.evalspec = PartPQEvalSpec(panoptic_part_eval_config['eval_spec_path'])
        # self.num_stuff_classes = 79
        # self.num_part_classes = 194
        self.ins2thing_ids = {i: tid for i, tid in enumerate(sorted(self.evalspec.eval_sid_things))}
        self.seg2stuff_ids = {i: tid for i, tid in enumerate(sorted(self.evalspec.eval_sid_stuff))}

        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.part_prefix = part_prefix

        data_infos = []
        img_lists = os.listdir(self.part_prefix)
        for img in img_lists:
            data_infos.append(
                {
                    'filename': img.replace('tif', 'jpg'),
                    'part_file': img,
                }
            )

        self.data_infos = data_infos
        if not test_mode:
            self._set_group_flag()

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def _panoptic2json(self, results, outfile_prefix):
        panoptic_json_results = []
        mmcv.mkdir_or_exist(outfile_prefix)
        for idx in range(len(self)):
            # img_id = self.img_ids[idx]
            panoptic = results[idx]
            png_string, segments_info = panoptic
            data = dict()
            for segment_info in segments_info:
                isthing = segment_info.pop('isthing')
                cat_id = segment_info['category_id']
                if isthing is True:
                    segment_info['category_id'] = self.ins2thing_ids[cat_id]
                else:
                    segment_info['category_id'] = self.seg2stuff_ids[cat_id]

            png_path = self.data_infos[idx]['filename'].replace('.jpg', '.png')
            # hack: to save all the images into one folder
            # png_path = png_path.split("/")[-1]
            png_save_path = osp.join(outfile_prefix, png_path)

            data['file_name'] = png_path
            data['image_id'] = png_path[:-4]
            # print(data['file_name'])
            # exit()
            with open(png_save_path, 'wb') as f:
                f.write(png_string)
            data['segments_info'] = segments_info
            panoptic_json_results.append(data)
        return panoptic_json_results

    def results2json(self, results, outfile_prefix):
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            if len(results[0]) == 3:  # dump the panoptic
                instance_segm_results = []
                panoptic_results = []
                for idx in range(len(self)):
                    det, seg, panoptic = results[idx]
                    instance_segm_results.append([det, seg])
                    panoptic_results.append(panoptic)
                panoptic_json = dict()
                panoptic_json['annotations'] = self._panoptic2json(
                    panoptic_results, outfile_prefix)
                result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
                mmcv.dump(panoptic_json, result_files['panoptic'])
            else:
                instance_segm_results = results
            # json_results = self._segm2json(instance_segm_results)
            # result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            # result_files['segm'] = f'{outfile_prefix}.segm.json'
            # mmcv.dump(json_results[0], result_files['bbox'])
            # mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    # def results2txt(self, results, outfile_prefix):
    #     """Dump the detection results to a txt file.
    #
    #     Args:
    #         results (list[list | tuple]): Testing results of the
    #             dataset.
    #         outfile_prefix (str): The filename prefix of the json files.
    #             If the prefix is "somepath/xxx",
    #             the txt files will be named "somepath/xxx.txt".
    #
    #     Returns:
    #         list[str]: Result txt files which contains corresponding \
    #             instance segmentation images.
    #     """
    #     try:
    #         import cityscapesscripts.helpers.labels as CSLabels
    #     except ImportError:
    #         raise ImportError('Please run "pip install citscapesscripts" to '
    #                           'install cityscapesscripts first.')
    #     result_files = []
    #     os.makedirs(outfile_prefix, exist_ok=True)
    #     prog_bar = mmcv.ProgressBar(len(self))
    #     for idx in range(len(self)):
    #         result = results[idx]
    #         filename = self.data_infos[idx]['filename']
    #         basename = osp.splitext(osp.basename(filename))[0]
    #         pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')
    #
    #         bbox_result, segm_result = result
    #         bboxes = np.vstack(bbox_result)
    #         # segm results
    #         if isinstance(segm_result, tuple):
    #             # Some detectors use different scores for bbox and mask,
    #             # like Mask Scoring R-CNN. Score of segm will be used instead
    #             # of bbox score.
    #             segms = mmcv.concat_list(segm_result[0])
    #             mask_score = segm_result[1]
    #         else:
    #             # use bbox score for mask score
    #             segms = mmcv.concat_list(segm_result)
    #             mask_score = [bbox[-1] for bbox in bboxes]
    #         labels = [
    #             np.full(bbox.shape[0], i, dtype=np.int32)
    #             for i, bbox in enumerate(bbox_result)
    #         ]
    #         labels = np.concatenate(labels)
    #
    #         assert len(bboxes) == len(segms) == len(labels)
    #         num_instances = len(bboxes)
    #         prog_bar.update()
    #         with open(pred_txt, 'w') as fout:
    #             for i in range(num_instances):
    #                 pred_class = labels[i]
    #                 classes = self.CLASSES[pred_class]
    #                 class_id = CSLabels.name2label[classes].id
    #                 score = mask_score[i]
    #                 mask = maskUtils.decode(segms[i]).astype(np.uint8)
    #                 png_filename = osp.join(outfile_prefix,
    #                                         basename + f'_{i}_{classes}.png')
    #                 mmcv.imwrite(mask, png_filename)
    #                 fout.write(f'{osp.basename(png_filename)} {class_id} '
    #                            f'{score}\n')
    #         result_files.append(pred_txt)
    #
    #     return result_files

    def format_results(self, results, jsonfile_prefix="./test", **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 outfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 metric_items=None):

        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]
        allowed_metrics = [
            'bbox', 'segm', 'cityscapes', 'panoptic', 'part'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, outfile_prefix)

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            if metric == 'part':
                from panoptic_parts.evaluation.eval_PartPQ import evaluate as pp_eval
                eval_spec_path = self.panoptic_part_eval_config.eval_spec_path
                basepath_gt = self.panoptic_part_eval_config.basepath_gt
                basepath_pred = self.panoptic_part_eval_config.basepath_pred
                images_json = self.panoptic_part_eval_config.images_json
                save_dir = self.panoptic_part_eval_config.save_dir
                res = pp_eval(eval_spec_path, basepath_gt, basepath_pred, images_json, save_dir)
                print_log(res[0], logger=logger)
                print_log(res[1], logger=logger)

            if metric == 'panoptic':
                from panopticapi.evaluation import pq_compute
                # print("pred folder", result_files['panoptic'].split('.')[0])
                with contextlib.redirect_stdout(io.StringIO()):
                    pq_res = pq_compute(
                        self.panoptic_gt_json,
                        result_files['panoptic'],
                        gt_folder=self.seg_prefix,
                        pred_folder=result_files['panoptic'].split('.')[0])
                results = parse_pq_results(pq_res)
                results['PQ_parts'] = sum([pq_res['per_class'][t]['pq'] for t in self.evalspec.eval_sid_parts]) / len(
                    self.evalspec.eval_sid_parts)
                results['PQ_noparts'] = sum(
                    [pq_res['per_class'][t]['pq'] for t in self.evalspec.eval_sid_no_parts]) / len(
                    self.evalspec.eval_sid_no_parts)
                for k, v in results.items():
                    eval_results[f'{metric}_{k}'] = f'{float(v):0.3f}'
                print_log(
                    'Panoptic Evaluation Results:\n' +
                    _print_panoptic_results(pq_res),
                    logger=logger)
                continue

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = None
        results['part_prefix'] = self.part_prefix
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['part_fields'] = []

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
