# PCWNet (ECCV 2022 oral)
This is the implementation of the paper PCW-Net: Pyramid Combination and Warping
Cost Volume for Stereo Matching, `ECCV 2022 oral`, Zhelun Shen, Yuchao Dai, Xibin Song, Zhibo Rao, Dingfu Zhou and Liangjun Zhang 

[comment]: <> ([\[Arxiv\]]&#40;https://arxiv.org/abs/2104.04314&#41;.)

Our method obtains the `1st` place on the stereo task of KITTI 2012 benchmark and `2nd` place on KITTI 2015 benchmark.

[comment]: <> (Camera ready version and supplementary Materials can be found in [\[CVPR official website\]]&#40;https://openaccess.thecvf.com/content/CVPR2021/html/Shen_CFNet_Cascade_and_Fused_Cost_Volume_for_Robust_Stereo_Matching_CVPR_2021_paper.html&#41;)

# Due to company policy, the code will be open sourced after approval is completed

[comment]: <> (## Abstract)

[comment]: <> (Recently, the ever-increasing capacity of large-scale annotated datasets has led to profound progress in stereo matching. However, most of these successes are limited to a specific dataset and cannot generalize well to other datasets. The main difficulties lie in the large domain differences and unbalanced disparity distribution across a variety of datasets, which greatly limit the real-world applicability of current deep stereo matching models. In this paper, we propose CFNet, a Cascade and Fused cost volume based network to improve the robustness of the stereo matching network. First, we propose a fused cost volume representation to deal with the large domain difference. By fusing multiple low-resolution dense cost volumes to enlarge the receptive field, we can extract robust structural representations for initial disparity estimation. Second, we propose a cascade cost volume representation to alleviate the unbalanced disparity distribution. Specifically, we employ a variance-based uncertainty estimation to adaptively adjust the next stage disparity search space, in this way driving the network progressively prune out the space of unlikely correspondences. By iteratively narrowing down the disparity search space and improving the cost volume resolution, the disparity estimation is gradually refined in a coarse-tofine manner. When trained on the same training images and evaluated on KITTI, ETH3D, and Middlebury datasets with the fixed model parameters and hyperparameters, our proposed method achieves the state-of-the-art overall performance and obtains the 1st place on the stereo task of Robust Vision Challenge 2020.)

[comment]: <> (# How to use)

[comment]: <> (## Environment)

[comment]: <> (* python 3.74)

[comment]: <> (* Pytorch == 1.1.0)

[comment]: <> (* Numpy == 1.15)

[comment]: <> (## Data Preparation)

[comment]: <> (Download [Scene Flow Datasets]&#40;https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html&#41;, [KITTI 2012]&#40;http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo&#41;, [KITTI 2015]&#40;http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo&#41;, [ETH3D]&#40;https://www.eth3d.net/&#41;, [Middlebury]&#40;https://vision.middlebury.edu/stereo/&#41;)


[comment]: <> (**KITTI2015/2012 SceneFlow**)

[comment]: <> (please place the dataset as described in `"./filenames"`, i.e., `"./filenames/sceneflow_train.txt"`, `"./filenames/sceneflow_test.txt"`, `"./filenames/kitticombine.txt"`)

[comment]: <> (**Middlebury/ETH3D**)

[comment]: <> (Our folder structure is as follows:)

[comment]: <> (```)

[comment]: <> (dataset)

[comment]: <> (├── KITTI2015)

[comment]: <> (├── KITTI2012)

[comment]: <> (├── Middlebury)

[comment]: <> (    │ ├── Adirondack)

[comment]: <> (    │   ├── im0.png)

[comment]: <> (    │   ├── im1.png)

[comment]: <> (    │   └── disp0GT.pfm)

[comment]: <> (├── ETH3D)

[comment]: <> (    │ ├── delivery_area_1l)

[comment]: <> (    │   ├── im0.png)

[comment]: <> (    │   ├── im1.png)

[comment]: <> (    │   └── disp0GT.pfm)

[comment]: <> (```)

[comment]: <> (Note that we use the full-resolution image of Middlebury for training as the additional training images don't have half-resolution version. We will down-sample the input image to half-resolution in the data argumentation. In contrast,  we use the half-resolution image and full-resolution disparity of Middlebury for testing. )

[comment]: <> (## Training)

[comment]: <> (**Scene Flow Datasets Pretraining**)

[comment]: <> (run the script `./scripts/sceneflow.sh` to pre-train on Scene Flow datsets. Please update `DATAPATH` in the bash file as your training data path.)

[comment]: <> (To repeat our pretraining details. You may need to replace the Mish activation function to Relu. Samples is shown in `./models/relu/`.)

[comment]: <> (**Finetuning**)

[comment]: <> (run the script `./scripts/robust.sh` to jointly finetune the pre-train model on four datasets,)

[comment]: <> (i.e., KITTI 2015, KITTI2012, ETH3D, and Middlebury. Please update `DATAPATH` and `--loadckpt` as your training data path and pretrained SceneFlow checkpoint file.)

[comment]: <> (## Evaluation)

[comment]: <> (**Joint Generalization**)

[comment]: <> (run the script `./scripts/eth3d_save.sh"`, `./scripts/mid_save.sh"` and `./scripts/kitti15_save.sh` to save png predictions on the test set of the ETH3D, Middlebury, and KITTI2015 datasets. Note that you may need to update the storage path of save_disp.py, i.e., `fn = os.path.join&#40;"/home3/raozhibo/jack/shenzhelun/cfnet/pre_picture/"`, fn.split&#40;'/'&#41;[-2]&#41;.)

[comment]: <> (**Corss-domain Generalization**)

[comment]: <> (run the script `./scripts/robust_test.sh"` to test the cross-domain generalizaiton of the model &#40;Table.3 of the main paper&#41;. Please update `--loadckpt` as pretrained SceneFlow checkpoint file.)

[comment]: <> (## Pretrained Models)

[comment]: <> ([Pretraining Model]&#40;https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view?usp=sharing&#41;)

[comment]: <> (You can use this checkpoint to reproduce the result we reported in Table.3 of the main paper)

[comment]: <> ([Finetuneing Moel]&#40;https://drive.google.com/file/d/1H6L-lQjF4yOxq23wxs3HW40B-0mLxUiI/view?usp=sharing&#41;)

[comment]: <> (You can use this checkpoint to reproduce the result we reported in the stereo task of Robust Vision Challenge 2020)

[comment]: <> (## Citation)

[comment]: <> (If you find this code useful in your research, please cite:)

[comment]: <> (```)

[comment]: <> (@InProceedings{Shen_2021_CVPR,)

[comment]: <> (    author    = {Shen, Zhelun and Dai, Yuchao and Rao, Zhibo},)

[comment]: <> (    title     = {CFNet: Cascade and Fused Cost Volume for Robust Stereo Matching},)

[comment]: <> (    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;CVPR&#41;},)

[comment]: <> (    month     = {June},)

[comment]: <> (    year      = {2021},)

[comment]: <> (    pages     = {13906-13915})

[comment]: <> (})

[comment]: <> (```)

[comment]: <> (# Acknowledgements)

[comment]: <> (Thanks to the excellent work GWCNet, Deeppruner, and HSMNet. Our work is inspired by these work and part of codes are migrated from [GWCNet]&#40;https://github.com/xy-guo/GwcNet&#41;, [DeepPruner]&#40;https://github.com/uber-research/DeepPruner/&#41; and [HSMNet]&#40;https://github.com/gengshan-y/high-res-stereo&#41;.)
