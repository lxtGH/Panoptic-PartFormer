# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from panoptic_parts.utils.visualization import experimental_colorize_label
from panoptic_parts.utils.format import decode_uids, encode_ids
from panoptic_parts.specs.dataset_spec import DatasetSpec
from panoptic_parts.specs.eval_spec import PartPQEvalSpec

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--datasetspec_path',default='panoptic_parts/specs/dataset_specs/cpp_datasetspec.yaml')
    parser.add_argument('--evalspec_path', default='panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    label_path = os.path.join('work_dirs/part_res', args.img.split('/')[-1].replace('leftImg8bit.png', 'gtFine_panoptic.png'))

    result = inference_detector(model, args.img)
    
    evalspec = PartPQEvalSpec(args.evalspec_path)
    spec = DatasetSpec(args.datasetspec_path)
    uids = np.array(Image.open(label_path), dtype=np.int32)
    sids, iids, pids = uids[..., 0], uids[..., 1], uids[..., 2]
    sids[sids==255] = 0
    pids[pids==255] = -1

    for x in evalspec.eval_sid_no_parts:
        pids[sids==x] = -1

    for x in evalspec.eval_sid_stuff:
        iids[sids==x] = -1

    uids = encode_ids(sids, iids, pids)
    uids_sem_inst_parts_colored = experimental_colorize_label(
            uids, sid2color=spec.sid2scene_color, emphasize_instance_boundaries=True,
            experimental_deltas=(60, 60, 60), experimental_alpha=0.5)
    img = Image.fromarray(uids_sem_inst_parts_colored)
    img.save(args.out_file)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)