#!/usr/bin/env bash
# source ~/.bashrc
# export PATH="/yrfs1/rc/zpchen/tools/anaconda9/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ssd/usr/local/cuda-9.2/lib64/
echo $LD_LIBRARY_PATH
date
bert_dir='/home/jiangwei/work/bert/chinese_roberta_wwm_ext_pytorch/'
python run_cail.py \
    --name train_v4 \
    --bert_model $bert_dir \
    --data_dir ../data/data_combine2019_1sentence \
    --batch_size 6 \
    --eval_batch_size 32 \
    --lr 1e-5 \
    --gradient_accumulation_steps 5 \
    --epochs 10 \
    --model_gpu 0 \

date

# run_cail.py parameters
# --name train_v1 --bert_model C:\Works\PretrainedModel\chinese_roberta_wwm_ext_pytorch --data_dir ../data --batch_size 2 --eval_batch_size 32 --lr 1e-5 --gradient_accumulation_steps 1 --epochs 10 --model_gpu -1
