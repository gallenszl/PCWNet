#!/usr/bin/env bash
set -x
# DATAPATH="/home2/dataset/scene_flow/"
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset sceneflow \
#     --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#     --epochs 1 --lr 0.001 --lrepochs "10,20:2" \
#     --model gwcnet-gc --logdir ./checkpoints/sceneflow_doubletrain/gwcnet-gc_mish25_test --test_batch_size 4 --batch_size 4 \
#     --loadckpt "/home3/raozhibo/jack/shenzhelun/gwc_multiple_refinement_test/gwc_multiple_refinement_test/checkpoints/sceneflow_doubletrain/gwcnet-gc_mish25/checkpoint_000023.ckpt"
DATAPATH=/mnt/shenzhelun/sceneflow
CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --lrepochs "10,12,14,16:2" --test_batch_size 2 --batch_size 2 \
    --model gwcnet-gc --logdir ./checkpoints/sceneflow/gwcnet-g