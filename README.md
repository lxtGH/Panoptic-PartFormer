# Panoptic-PartFormer ECCV-2022 [[Introduction Video](https://github.com/lxtGH/Panoptic-PartFormer/raw/main/video_poster/5523.mp4)], [[Poster](https://github.com/lxtGH/Panoptic-PartFormer/raw/main/video_poster/ECCV22_poster_5523.pdf)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/panoptic-partformer-learning-a-unified-model/part-aware-panoptic-segmentation-on)](https://paperswithcode.com/sota/part-aware-panoptic-segmentation-on?p=panoptic-partformer-learning-a-unified-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/panoptic-partformer-learning-a-unified-model/part-aware-panoptic-segmentation-on-pascal)](https://paperswithcode.com/sota/part-aware-panoptic-segmentation-on-pascal?p=panoptic-partformer-learning-a-unified-model)

[Xiangtai Li](https://lxtgh.github.io/),
Shilin Xu,
[Yibo Yang](https://scholar.google.com/citations?user=DxXXnCcAAAAJ&hl=zh-CN), 
[Guangliang Cheng](https://scholar.google.com/citations?user=FToOC-wAAAAJ),
[Yunhai Tong](https://eecs.pku.edu.cn/info/1475/9689.htm),
[Dacheng Tao](http://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=zh-CN).

Source Code and model of our ECCV-2022 paper. 
*Our re-implementation achieve slightly better results than original submitted paper.*


## Introduction

![Figure](./figs/ppformer_teaser.png)

Panoptic Part Segmentation (PPS) aims to unify panoptic
segmentation and part segmentation into one task. Previous work mainly
utilizes separated approaches to handle thing, stuff, and part predictions
individually without performing any shared computation and task association.
In this work, we aim to unify these tasks at the architectural
level, designing the first end-to-end unified method named Panoptic-
PartFormer. 

We model things, stuff, and part as object queries and directly
learn to optimize the all three predictions as unified mask prediction
and classification problem. We design a decoupled decoder to
generate part feature and thing/stuff feature respectively

## Installation

It requires the following OpenMMLab packages:

- MMCV-full == v1.3.18
- MMDetection == v2.18.0
- panopticapi

The packages that in the requirement.txt file.

## Usage

### Data preparation

The basic data formats are as following:

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection). 
The data structure looks like below:

For Cityscapes Panoptic Part (CPP) and PascalContext Panoptic Part (PPP) dataset:

The Extra Anotations for PPP dataset can be downloaded [here](https://drive.google.com/drive/folders/1dXi-K1L18GKwQedfuR9PSPU5mNCFsTBb?usp=sharing).

```
data/
├── VOCdevkit # pascal context part dataset 
│   ├── gt_panoptic
│   ├── labels_57part
│   ├── VOC2010
│   │   ├── JPEGImages
├── cityscapes # cityscape pascal part dataset 
│   ├── annotations
│   │   ├── cityscapes_panoptic_{train,val}.json
│   │   ├── instancesonly_filtered_gtFine_{train,val}.json
│   │   ├── val_images.json
│   ├── gtFine
│   ├── gtFinePanopticParts
│   ├── leftImg8bit
```
### Pretrained Models:

#### Cityscapes Pretrained Model

Note that The cityscapes model results can be improved via large crop finetuning 
see the config city_r50_fam_ft.py.


R-50: [link](https://1drv.ms/u/s!Ai4mxaXd6lVBfdSc09Z-Wkkkp8E?e=RKBA4f)

Swin-base: [link](https://1drv.ms/u/s!Ai4mxaXd6lVBfwlSs4KjN5go9zA?e=gIVT2b)

#### Pascal Context Pretrained Model on COCO

R-50: [link](https://1drv.ms/u/s!Ai4mxaXd6lVBfG0R8Nj2TcYry7w?e=grw4Ui)

Swin-base: [link](https://1drv.ms/u/s!Ai4mxaXd6lVBfpWmnRmcl7lAB0k?e=2acus0)


### Trained Models

### CPP

R-50 [link](https://1drv.ms/u/s!Ai4mxaXd6lVBgQIEinLVprQyAf0J?e=0ulzUr), PartPQ: 57.5

Swin-base [link](https://1drv.ms/u/s!Ai4mxaXd6lVBdNL9EzFbpUc5N6I?e=mmxa5Z) PartPQ: 61.9

### PPP

R-50 [link](https://1drv.ms/u/s!Ai4mxaXd6lVBgQAS47KqSrrMrXAV?e=VoS1Ge) PartPQ: 38.1

R-101 [link](https://1drv.ms/u/s!Ai4mxaXd6lVBgQMjRczqr5xgLxZ9?e=x3ZvE2) PartPQ: 40.2

Swin-base [link](https://1drv.ms/u/s!Ai4mxaXd6lVBgQHfYk6BNkDIU-Qh?e=4LNnVq) PartPQ: 47.8 


### Training and testing
For single machine with multi gpus. 
To reproduce the performance.
Make sure you have loaded the ckpt correctly. 

```bash
# train
sh ./tools/dist_train.sh $CONFIG $NUM_GPU
# test
sh ./tools/dist_test.sh $CONFIG $CHECKPOINT --eval panoptic part
```

Note for ppp dataset training, better to add --no-validate flag since the long evaluation period 
and more cpu/memory cost. And then eval the model in an offline manner. 

for multi machines, we use *slurm*

```bash
sh ./tools/slurm_train.sh $PARTITION $JOB_NAME $CONFIG $WORK_DIR

sh ./tools/slurm_test.sh $PARTITION $JOB_NAME $CONFIG $CHECKPOINT --eval panoptic, part
```

- PARTITION: the slurm partition you are using
- CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CONFIG: the config files under the directory `configs/`
- JOB_NAME: the name of the job that are necessary for slurm

You can also run training and testing without slurm by directly mim for instance/semantic segmentation or `tools/train.py` for panoptic segmentation like below:

### Demo Visualization
for cityscapes panoptic part
```bash
python demo/image_demo.py $IMG_FILE $CONFIG $CHECKPOING $OUT_FILE
```
for pascal panoptic part
```bash
python demo/image_demo.py $IMG_FILE $CONFIG $CHECKPOING $OUT_FILE --datasetspec_path=$1 --evalspec_path=$2
```

### Visual Results

![Figure](./figs/panoptic_part_former_vis_res.png)

### Acknowledgement

We build our codebase based on K-Net and mmdetection. Much thanks for their open-sourced code.
Mainly refer to the implementation of thing/stuff kernels (query) interaction part. 


### Citation
If you find this repo is useful for your research, Please consider citing our paper:

```
@article{li2022panoptic,
  title={Panoptic-partformer: Learning a unified model for panoptic part segmentation},
  author={Li, Xiangtai and Xu, Shilin and Cheng, Yibo Yang and Tong, Yunhai and Tao, Dacheng and others},
  journal={ECCV},
  year={2022}
}
```
### License

MIT license 
