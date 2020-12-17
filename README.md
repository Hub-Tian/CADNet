# CADet: Context-Aware Dynamic Feature Extraction for 3D Object Detection in Point Clouds 

`CADet` is an one-stage 3D object detector proposed to handle the density variance  in point cloud. We integrate our method into the awesome codebase `PCDet`. For more details of our work, please refer our paper https://arxiv.org/abs/1912.04775v3

## AP on KITTI Dataset

```
Car AP_R11@0.70,0.70,0.70:
bbox AP:90.82,89.71,88.15
bev  AP:90.28,87.11,83.92
3d   AP:88.51,78.20,75.74
aos  AP:90.80,89.48,87.80

Car AP_R40@0.70,0.70,0.70:
bbox AP:95.52,92.13,90.66
bev  AP:92.55,88.22,86.33
3d   AP:88.84,79.43,75.95
aos  AP:95.49,91.86,90.27
```



## Installation

The installation is following the steps in `pcdet`.
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.1 or higher
* CUDA 9.0 or higher

### Install `cadet`
1. Clone this repository.
```shell
git clone https://github.com/Hub-Tian/CADNet.git
```

2. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we extended the implementation from [`spconv`](https://github.com/traveller59/spconv). 

```
cd spconv
python setup.py bdist_wheel
cd ../dist
pip install ./spconv*
```


3. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```

## Dataset Preparation

Currently we only support KITTI dataset, and contributions are welcomed to support more datasets.

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [here](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training): 

```
PCDet
├── data
│   ├── kitti
│   │   │──ImageSets
│   │   │──training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │──testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command in the path `pcdet/datasets/kitti`: 
```python 
python kitti_dataset.py create_kitti_infos
```

## Getting Started
All the config files are within `tools/cfgs/`. 

### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file /tools/cfgs/pointpillar_expand_car.yaml --batch_size 4 --ckpt ${CKPT}
```

* To evaluate all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file /tools/cfgs/pointpillar_expand_car.yaml --batch_size 4 --eval_all
```


### Train a model
* Train with multiple GPUs:
```shell script
bash scripts/dist_train.sh ${NUM_GPUS} \ 
    --cfg_file /tools/cfgs/pointpillar_expand_car.yaml --batch_size ${BATCH_SIZE}
```

* Train with multiple machines:
```shell script
bash scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} \ 
    --cfg_file /tools/cfgs/pointpillar_expand_car.yaml --batch_size ${BATCH_SIZE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file /tools/cfgs/pointpillar_expand_car.yaml --batch_size ${BATCH_SIZE}
```

## Acknowledgement
This repo is based on `pcdet`(https://github.com/open-mmlab/OpenPCDet).



## Citation 
If you find this work useful in your research, please consider cite:
```
@article{tian2019context,
  title={Context-Aware Dynamic Feature Extraction for 3D Object Detection in Point Clouds},
  author={Tian, Yonglin and Huang, Lichao and Yu, Hui and Wu, Xiangbin and Li, Xuesong and Wang, Kunfeng and Wang, Zilei and Wang, Fei-Yue},
  journal={arXiv preprint  arXiv:1912.04775v3},
  year={2020}
}
```
