#!/usr/bin/env bash
set -x
DATAPATH="/mnt/shenzhelun/dataset/Kitti2012"
CUDA_VISIBLE_DEVICES=1 python3.7 save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti12_test.txt --model gwcnet-gc \
--loadckpt "/mnt/shenzhelun/pcwnet_github/checkpoint_000294.ckpt"