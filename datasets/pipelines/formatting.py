import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.pipelines.formating import DefaultFormatBundle


@PIPELINES.register_module()
class PartDefaultFormatBundle(DefaultFormatBundle):
    def __call__(self, results):
        super(PartDefaultFormatBundle, self).__call__(results)
        if 'gt_part' in results:
            results['gt_part'] = DC(
                to_tensor(np.ascontiguousarray(results['gt_part'][None, ...])), stack=True)
        return results