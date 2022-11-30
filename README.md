# PCWNet (ECCV 2022 oral)
This is the official pytorch implementation of the paper PCW-Net: Pyramid Combination and Warping
Cost Volume for Stereo Matching, `ECCV 2022 oral`, Zhelun Shen, Yuchao Dai, Xibin Song, Zhibo Rao, Dingfu Zhou and Liangjun Zhang 

[comment]: <> ([\[Arxiv\]]&#40;https://arxiv.org/abs/2104.04314&#41;.)

**Our method obtains the `1st` place on the stereo task of KITTI 2012 benchmark and `2nd` place on KITTI 2015 benchmark.**

**Note : see the paddle implementation and the awesome unified framework for stereo matching in [Paddledepth](https://github.com/PaddlePaddle/PaddleDepth)**

[comment]: <> (Camera ready version and supplementary Materials can be found in [\[CVPR official website\]]&#40;https://openaccess.thecvf.com/content/CVPR2021/html/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.html&#41;)

## Abstract
Existing deep learning based stereo matching methods either focus on 
achieving optimal performances on the target dataset while with poor generalization for other datasets 
or focus on handling the cross-domain generalization 
by suppressing the domain sensitive features which results in a significant sacrifice on the performance. 
To tackle these problems, we propose PCW-Net, a Pyramid Combination and Warping cost volume-based network 
to achieve good performance on both cross-domain generalization and stereo matching 
accuracy on various benchmarks. 
In particular, our PCW-Net is designed for two purposes. First, we construct combination volumes 
on the upper levels of the pyramid and develop a cost volume fusion module to integrate them 
for initial disparity estimation. Multi-scale receptive fields can be covered by 
fusing multi-scale combination volumes, thus, domain-invariant features can be extracted. 
Second, we construct the warping volume at the last level of the pyramid for disparity refinement. 
The proposed warping volume can narrow down the 
residue searching range from the initial disparity searching range to a fine-grained one, 
which can dramatically alleviate the difficulty of the network to 
find the correct residue in an unconstrained residue searching space. 
When training on synthetic datasets and generalizing to unseen real datasets, 
our method shows strong cross-domain generalization and outperforms 
existing state-of-the-arts with a large margin. After fine-tuning on the real datasets, 
our method ranks first on KITTI 2012, second on KITTI 2015, and first on the Argoverse among all published methods as of 7, March 2022.

# How to use
## Environment
* python 3.74
* Pytorch == 1.1.0
* Numpy == 1.15
## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), [ETH3D](https://www.eth3d.net/), [Middlebury](https://vision.middlebury.edu/stereo/)

**KITTI2015/2012 SceneFlow**

please place the dataset as described in `"./filenames"`, i.e., `"./filenames/sceneflow_train.txt"`, `"./filenames/sceneflow_test.txt"`, `"./filenames/kitticombine.txt"`

**Middlebury/ETH3D**

Our folder structure is as follows:
```
dataset
├── KITTI2015
├── KITTI2012
├── Middlebury
    │ ├── Adirondack
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
├── ETH3D
    │ ├── delivery_area_1l
    │   ├── im0.png
    │   ├── im1.png
    │   └── disp0GT.pfm
```
Note that we use the half-resolution dataset of Middlebury for testing. 
## Training
**Scene Flow Datasets Pretraining**

run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.
To repeat our pretraining details. You may need to replace the Mish activation function to Relu.  Samples are shown in `./models/relu/`.

**Finetuning**

run the script `./scripts/kitti15.sh` and `./scripts/kitti12.sh` to finetune our pretraining model on the KITTI dataset. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.
## Evaluation
**Corss-domain Generalization**

run the script `./scripts/generalization_test.sh"` to test the cross-domain generalizaiton of the model (Table.2 of the main paper). Please update `--loadckpt` as pretrained SceneFlow checkpoint file.

**Finetuning Performance**

run the script `./scripts/kitti15_save.sh"` and `./scripts/kitti12_save.sh"` to generate the corresponding test images of KITTI 2015&2012

## Pretrained Models

[Sceneflow Pretraining Model](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view?usp=sharing)

You can use this checkpoint to reproduce the result we reported in Table.2 of the main paper

[KITTI 2012 Finetuneing Moel](https://drive.google.com/file/d/14MANgQJ15Qzukv9SoL9MYobg5xUjE-u0/view?usp=sharing)

You can use this checkpoint to reproduce the result we submitted on KITTI 2012 benchmark.

## Citation

If you find this code useful in your research, please cite:

```
@inproceedings{shen2022pcw,
  title={PCW-Net: Pyramid Combination and Warping Cost Volume for Stereo Matching},
  author={Shen, Zhelun and Dai, Yuchao and Song, Xibin and Rao, Zhibo and Zhou, Dingfu and Zhang, Liangjun},
  booktitle={European Conference on Computer Vision},
  pages={280--297},
  year={2022},
  organization={Springer}
}

```

# Acknowledgements

Thanks to the excellent work GWCNet and HSMNet. Our work is inspired by these work and part of codes are migrated from [GWCNet](https://github.com/xy-guo/GwcNet) and [HSMNet](https://github.com/gengshan-y/high-res-stereo).
