# !/usr/bin python
# -*- coding:utf-8 -*-
import os
import sys
import gzip
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.reading_comprehension_net import ReadingComprehensionModel
from reading_comprehension.data_process import InputFeatures, Example


if __name__ == '__main__':

    RCModel = ReadingComprehensionModel(config='configs/reading_comprehension_config.yml')

    RCModel.load_model(r'runs/rc_20200701-211818', 'L')

    test_set_feature_file = '../data/dev_feature.pkl.gz'
    with gzip.open(test_set_feature_file, 'rb') as f:
        test_set_features = pickle.load(f)

    # print('missing support fact ids:')
    # for example in test_set_features:
    #     if example.ans_type != 3 and len(example.sup_fact_ids) == 0:
    #         print(example.qas_id)

    (val_span_loss, val_answer_type_loss, val_support_fact_loss, val_loss, val_answer_type_accu,
     val_span_iou, val_answer_score, val_support_fact_accu, val_support_fact_recall,
     val_support_fact_precision, val_support_fact_f1, val_joint_metric
     ) = RCModel.test(test_set=test_set_features, test_batch_size=2)

    print('Test Done.')
