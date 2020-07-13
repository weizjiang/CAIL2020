import argparse
import os
from tqdm import tqdm
import json
import gzip
import pickle
from transformers import BertModel
from transformers import BertConfig as BC
from transformers import BertTokenizer
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper
from data_process import InputFeatures, Example, read_examples, convert_examples_to_features

import torch
from torch import nn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()

    for batch in tqdm(dataloader):

        batch['context_mask'] = batch['context_mask'].float()
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict={}
    for key,value in answer_dict.items():
        new_answer_dict[key]=value.replace(" ","")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)


if __name__ == '__main__':
    infile = "../input/data.json"
    outfile = "../result/result.json"

    parser = argparse.ArgumentParser()
    args = set_config()

    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu == 0:
        # reset 'model_gpu' in case no gpu avaliable
        args.model_gpu = '-1'

    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    tokenizer = BertTokenizer.from_pretrained('./data')
    examples = read_examples(full_file=infile)
    with gzip.open('./data/dev_example.pkl.gz', 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open('./data/dev_feature.pkl.gz', 'wb') as fout:
        pickle.dump(features, fout)

    # args.data_dir default is './data'
    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader

    roberta_config = BC.from_pretrained('./data')
    args.input_dim = roberta_config.hidden_size
    encoder = BertModel(roberta_config)
    model = BertSupportNet(config=args, encoder=encoder)
    model.load_state_dict(torch.load('./checkpoints/ckpt_seed_82_epoch_10_99999.pth',
                                     map_location=torch.device('cpu')))
    if args.model_gpu != '-1':
        model.to('cuda')

    model = torch.nn.DataParallel(model)

    predict(model, eval_dataset, dev_example_dict, dev_feature_dict, os.path.abspath(outfile))

