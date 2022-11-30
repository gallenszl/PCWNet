#!/usr/bin/env bash
set -x
DATAPATH=/mnt/shenzhelun/dataset/
CUDA_VISIBLE_DEVICES=1,2 python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitticombine.txt --testlist ./filenames/kitti15_errortest.txt \
    --epochs 300 --lr 0.001 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti15/test \
    --test_batch_size 2 --batch_size 2