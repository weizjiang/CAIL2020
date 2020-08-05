#!/usr/bin/env bash
source ~/.bashrc
# export PATH="/yrfs1/rc/zpchen/tools/anaconda9/bin:$PATH"
INPUT_TRAIN_FILE=$1/train.json
INPUT_DEV_FILE=$1/dev.json

OUTPUT_DIR=$2 #this dir must the same as the data_dir in train.sh

mkdir ${OUTPUT_DIR}
tokenizer_path='/home/jiangwei/work/bert/chinese_roberta_wwm_ext_pytorch/'

python data_process.py \
    --tokenizer_path=$tokenizer_path \
    --full_data=${INPUT_TRAIN_FILE} \
    --example_output=${OUTPUT_DIR}/train_example.pkl.gz \
    --feature_output=${OUTPUT_DIR}/train_features \

python data_process.py \
    --tokenizer_path=$tokenizer_path \
    --full_data=${INPUT_DEV_FILE} \
    --example_output=${OUTPUT_DIR}/dev_example.pkl.gz \
    --feature_output=${OUTPUT_DIR}/dev_feature.pkl.gz \


# data_process.py parameters

# # train set
# --tokenizer_path=C:\Works\PretrainedModel\chinese_roberta_wwm_ext_pytorch --full_data=C:\Works\Code\CAIL2020\ydlj\data\train_big.json --example_output=../data/train_example.pkl.gz --feature_output=../data/train_feature.pkl.gz

# # dev set
# --tokenizer_path=C:\Works\PretrainedModel\chinese_roberta_wwm_ext_pytorch --full_data=C:\Works\Code\CAIL2020\ydlj\data\dev_big.json --example_output=../data/dev_example.pkl.gz --feature_output=../data/dev_feature.pkl.gz
