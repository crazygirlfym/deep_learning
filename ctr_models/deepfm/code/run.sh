#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
python DeepFM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 100  --lr 0.1
