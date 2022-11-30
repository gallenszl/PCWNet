#!/usr/bin/env bash
set -x
# DATAPATH=/mnt/shenzhelun/dataset/Kitti2012
# CUDA_VISIBLE_DEVICES=1,2,3,4 python3.7 generalization_test.py --dataset kitti \
#     --datapath $DATAPATH --trainlist ./filenames/kitti12_all.txt \
#     --testlist ./filenames/kitti12_all.txt \
#     --epochs 1  --lr 0.001 --batch_size 2 --lrepochs "150:10" \
#     --model gwcnet-gc --logdir ./checkpoints/kitti15/test \
#     --test_batch_size 1 \
#     --loadckpt /mnt/shenzhelun/pcwnet_github/sceneflow_pretrain.ckpt
DATAPATH="/mnt/shenzhelun/dataset"
CUDA_VISIBLE_DEVICES=1,2,3,4 python3.7 generalization_test.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt \
    --testlist ./filenames/kitti15_errortest.txt \
    --epochs 1  --lr 0.001 --batch_size 2 --lrepochs "150:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti15/test \
    --test_batch_size 1 \
    --loadckpt /mnt/shenzhelun/pcwnet_github/sceneflow_pretrain.ckpt