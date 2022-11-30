#!/usr/bin/env bash
set -x
DATAPATH="/data/Documents/Database/robust/kitti12_15/Kitti/"
CUDA_VISIBLE_DEVICES=2 python3.7 save_disp.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --model gwcnet-gc --loadckpt "/data2/jack/shenzhelun/gwc_multiple_refinement_test/gwc_multiple_refinement_test/checkpoints/kitti15/gwcnet-gc_200_200_mishall/checkpoint_000253.ckpt"