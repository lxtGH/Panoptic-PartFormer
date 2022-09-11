import contextlib
import io
import os
import glob
import tempfile
import logging
import os.path as osp
from collections import OrderedDict
from pathlib import Path
import pycocotools.mask as maskUtils
from PIL import Image
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval
from datasets.coco_panoptic import parse_pq_results, _print_panoptic_results
from mmdet.utils import get_root_logger
from panoptic_parts.specs.dataset_spec import DatasetSpec
from panoptic_parts.specs.eval_spec import PartPQEvalSpec


@DATASETS.register_module()
class CityscapesPanopticPartDataset(CocoDataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
    PART_CLASSES = (
        'person-torso', 'person-head', 'person-arm', 'person-leg',
        'rider-torso', 'rider-head', 'rider-arm', 'rider-leg',
        'car-window', 'car-wheel', 'car-light', 'car-license plate', 'car-chassis',
        'truck-window', 'truck-wheel', 'truck-light', 'truck-license plate', 'truck-chassis',
        'bus-window', 'bus-wheel', 'bus-light', 'bus-license plate', 'bus-chassis'
    )

    def __init__(self, part_prefix=None, panoptic_part_eval_config=None, **kwargs):
        super(CityscapesPanopticPartDataset, self).__init__(**kwargs)
        self.part_prefix = part_prefix
        self.panoptic_part_eval_config = panoptic_part_eval_config
        self.id2label = {sid: i for i, sid in enumerate(self.stuff_ids)}
        self.id2label.update({tid: i + 11 for i, tid in enumerate(self.cat_ids)})
        self.logger = get_root_logger()
        self.evalspec = PartPQEvalSpec(self.panoptic_part_eval_config['eval_spec_path'])
        print(self.panoptic_part_eval_config)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file['ins_ann'])
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = sorted(self.coco.get_img_ids())

        self.panoptic_anns = mmcv.load(ann_file['panoptic_ann'])

        self.part_ids = [list(range(len(self.PART_CLASSES)))]
        self.part2part_ids = {name: i for i, name in enumerate(self.PART_CLASSES)}
        self.part_ids2part = {i: name for i, name in enumerate(self.PART_CLASSES)}
        self.stuff_ids = [
            k['id'] for k in self.panoptic_anns['categories']
            if k['isthing'] == 0
        ]

        self.thing_ids = [
            k['id'] for k in self.panoptic_anns['categories']
            if k['isthing'] == 1
        ]

        assert self.thing_ids == self.cat_ids

        self.seg2stuff_ids = {
            i: stuff_id
            for i, stuff_id in enumerate(self.stuff_ids)
        }

        self.ins2thing_ids = {
            i: thing_id
            for i, thing_id in enumerate(self.thing_ids)
        }

        self.vps = ann_file.get("vps", None)

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['part_file'] = info['filename'].replace('leftImg8bit.png', 'gtFinePanopticParts.tif')
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['part_prefix'] = self.part_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['part_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        # hack for no part objects
        no_part_imgs = ['aachen/aachen_000173_000019_leftImg8bit.png',
                        'weimar/weimar_000097_000019_leftImg8bit.png',
                        'weimar/weimar_000044_000019_leftImg8bit.png',
                        'weimar/weimar_000067_000019_leftImg8bit.png',
                        'tubingen/tubingen_000067_000019_leftImg8bit.png',
                        'tubingen/tubingen_000142_000019_leftImg8bit.png',
                        'strasbourg/strasbourg_000000_035571_leftImg8bit.png',
                        'strasbourg/strasbourg_000000_023854_leftImg8bit.png',
                        'strasbourg/strasbourg_000000_012934_leftImg8bit.png',
                        'strasbourg/strasbourg_000000_036016_leftImg8bit.png',
                        'dusseldorf/dusseldorf_000106_000019_leftImg8bit.png',
                        'dusseldorf/dusseldorf_000101_000019_leftImg8bit.png',
                        'bochum/bochum_000000_031152_leftImg8bit.png',
                        'monchengladbach/monchengladbach_000000_015561_leftImg8bit.png']
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_in_cat
                                         or all_iscrowd):
                continue
            if img_info['file_name'] in no_part_imgs:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            img_info (dict): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, \
                bboxes_ignore, labels, masks, seg_map. \
                "masks" are already decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=img_info['segm_file'],
            part_map=img_info['part_file']
        )

        return ann

    def _panoptic2json(self, results, outfile_prefix):
        panoptic_json_results = []
        mmcv.mkdir_or_exist(outfile_prefix)
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            panoptic = results[idx]
            png_string, segments_info = panoptic
            data = dict()
            # hack
            # To match the corresponding ids for panoptic segmentation prediction
            # for both cityscape vps and cityscapes
            if self.vps is not None:
                data['image_id'] = "_".join(self.data_infos[idx]['file_name'].split(".")[0].split("_")[:5])
                # print(data['image_id'])
                # exit()
            else:
                data['image_id'] = self.data_infos[idx]['file_name'].split("/")[-1].split(".")[0][:-12]

            for segment_info in segments_info:
                isthing = segment_info.pop('isthing')
                cat_id = segment_info['category_id']
                if isthing is True:
                    segment_info['category_id'] = self.ins2thing_ids[cat_id]
                else:
                    segment_info['category_id'] = self.seg2stuff_ids[cat_id]

            png_path = self.data_infos[idx]['file_name'].replace(
                '.jpg', '.png')
            # hack: to save all the images into one folder
            png_path = png_path.split("/")[-1]
            png_save_path = osp.join(outfile_prefix, png_path)

            data['file_name'] = png_path
            # print(data['file_name'])
            # exit()
            with open(png_save_path, 'wb') as f:
                f.write(png_string)
            data['segments_info'] = segments_info
            panoptic_json_results.append(data)
        return panoptic_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
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
            json_results = self._segm2json(instance_segm_results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def results2txt(self, results, outfile_prefix):
        """Dump the detection results to a txt file.

        Args:
            results (list[list | tuple]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx",
                the txt files will be named "somepath/xxx.txt".

        Returns:
            list[str]: Result txt files which contains corresponding \
                instance segmentation images.
        """
        try:
            import cityscapesscripts.helpers.labels as CSLabels
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        result_files = []
        os.makedirs(outfile_prefix, exist_ok=True)
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.data_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')

            bbox_result, segm_result = result
            bboxes = np.vstack(bbox_result)
            # segm results
            if isinstance(segm_result, tuple):
                # Some detectors use different scores for bbox and mask,
                # like Mask Scoring R-CNN. Score of segm will be used instead
                # of bbox score.
                segms = mmcv.concat_list(segm_result[0])
                mask_score = segm_result[1]
            else:
                # use bbox score for mask score
                segms = mmcv.concat_list(segm_result)
                mask_score = [bbox[-1] for bbox in bboxes]
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            assert len(bboxes) == len(segms) == len(labels)
            num_instances = len(bboxes)
            prog_bar.update()
            with open(pred_txt, 'w') as fout:
                for i in range(num_instances):
                    pred_class = labels[i]
                    classes = self.CLASSES[pred_class]
                    class_id = CSLabels.name2label[classes].id
                    score = mask_score[i]
                    mask = maskUtils.decode(segms[i]).astype(np.uint8)
                    png_filename = osp.join(outfile_prefix,
                                            basename + f'_{i}_{classes}.png')
                    mmcv.imwrite(mask, png_filename)
                    fout.write(f'{osp.basename(png_filename)} {class_id} '
                               f'{score}\n')
            result_files.append(pred_txt)

        return result_files

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
        """Evaluation in Cityscapes/COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            outfile_prefix (str | None): The prefix of output file. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with COCO protocol, it would be the
                prefix of output json file. For example, the metric is 'bbox'
                and 'segm', then json files would be "a/b/prefix.bbox.json" and
                "a/b/prefix.segm.json".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output txt/png files. The output files would be
                png images under folder "a/b/prefix/xxx/" and the file name of
                images would be written into a txt file
                "a/b/prefix/xxx_pred.txt", where "xxx" is the video_poster name of
                cityscapes. If not specified, a temp file will be created.
                Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: COCO style evaluation metric or cityscapes mAP \
                and AP@50.
        """
        eval_results = dict()

        metrics = metric.copy() if isinstance(metric, list) else [metric]
        allowed_metrics = [
            'bbox', 'segm', 'cityscapes', 'panoptic', 'part'
        ]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, outfile_prefix, logger))
            metrics.remove('cityscapes')

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, outfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        # self.eval_segm_iou(self.panoptic_part_eval_config.basepath_pred, self.seg_prefix)
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
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'panoptic':
                from panopticapi.evaluation import pq_compute
                # print("pred folder", result_files['panoptic'].split('.')[0])
                with contextlib.redirect_stdout(io.StringIO()):
                    pq_res = pq_compute(
                        self.ann_file['panoptic_ann'],
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

    def eval_segm_iou(self, pred_dir, gt_dir):
        n = 20
        gt_dir = Path(gt_dir).parent / 'val'
        pred_dir = Path(pred_dir)
        preds = [x for x in pred_dir.iterdir() if x.is_file()]
        gts = [x.name.replace('panoptic', 'labelTrainIds') for x in preds]
        gts = [gt_dir / x.split('_')[0] / x for x in gts]
        hist = np.zeros((n, n))
        for i in range(len(preds)):
            p = np.array(Image.open(preds[i]))[:, :, 0]
            for k, v in self.id2label.items():
                p[p == k] = v
            p[p == 255] = 19
            t = np.array(Image.open(gts[i]))
            t[t == 255] = 19
            k = (t >= 0) & (t < n)
            hist += np.bincount(n * t[k].astype(int) + p[k], minlength=n ** 2).reshape(n, n)
        iu = hist.diagonal() / (hist.sum(0) + hist.sum(1) - hist.diagonal())
        print('segm_iou---', iu[:-1])
        print('miou---', np.nanmean(iu[:-1]))

    def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of output txt file
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: Cityscapes evaluation results, contains 'mAP' \
                and 'AP@50'.
        """

        try:
            import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install citscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, txtfile_prefix)

        if tmp_dir is None:
            result_dir = osp.join(txtfile_prefix, 'results')
        else:
            result_dir = osp.join(tmp_dir.name, 'results')

        eval_results = OrderedDict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        # set global states in cityscapes evaluation API
        CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
        CSEval.args.predictionPath = os.path.abspath(result_dir)
        CSEval.args.predictionWalk = None
        CSEval.args.JSONOutput = False
        CSEval.args.colorized = False
        CSEval.args.gtInstancesFile = os.path.join(result_dir,
                                                   'gtInstances.json')
        CSEval.args.groundTruthSearch = os.path.join(
            self.img_prefix.replace('leftImg8bit', 'gtFine'),
            '*/*_gtFine_instanceIds.png')

        groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
        assert len(groundTruthImgList), 'Cannot find ground truth images' \
                                        f' in {CSEval.args.groundTruthSearch}.'
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
        CSEval_results = CSEval.evaluateImgLists(predictionImgList,
                                                 groundTruthImgList,
                                                 CSEval.args)['averages']

        eval_results['mAP'] = CSEval_results['allAp']
        eval_results['AP@50'] = CSEval_results['allAp50%']
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
