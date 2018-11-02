#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 \
#python FM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 256  --lr 0.01 --keep 0.7 --num_gpus=2
python AFM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor [8,256] --keep [1.0,0.5] --lamda_attention 2.0 --lr 0.1
