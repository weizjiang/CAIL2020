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

    # RCModel.load_model(r'runs/rc_20200701-211818', 'L')
    # RCModel.load_model(r'runs/rc_20200702-144524', 'L')
    # RCModel.load_model(r'runs/rc_20200707-193921', 'B')
    # RCModel.load_model(r'runs/rc_20200707-204721', 'B')
    # RCModel.load_model(r'runs/rc_20200708-211940', 'B')
    # RCModel.load_model(r'runs/rc_20200708-212137', 'B')
    # RCModel.load_model(r'runs/rc_20200709-160717', 'B')
    # RCModel.load_model(r'runs/rc_20200709-163056', 'B')
    # RCModel.load_model(r'runs/rc_20200710-120216', 'B')
    # RCModel.load_model(r'runs/rc_20200710-120452', 'B')
    # RCModel.load_model(r'runs/rc_20200713-131312', 'B')
    # RCModel.load_model(r'runs/rc_20200713-151448', 'B')
    # RCModel.load_model(r'runs/rc_20200714-202143', 'B')
    RCModel.load_model(r'runs/rc_20200715-172519', 'B')
    # RCModel.load_model(r'runs/rc_20200715-193558', 'B')

    # test_set_feature_file = '../data/dev_feature.pkl.gz'
    # test_set_feature_file = '../data/train_feature.pkl.gz'
    # test_set_feature_file = '../data/data_combine2019_1sentence/train_feature.pkl.gz'
    test_set_feature_file = '../data/data_correct_span/dev_feature.pkl.gz'
    
    with gzip.open(test_set_feature_file, 'rb') as f:
        test_set_features = pickle.load(f)

    print('missing support fact ids:')
    for example in test_set_features:
        if example.ans_type != 3 and len(example.sup_fact_ids) == 0:
            print(example.qas_id)

    print('missing answer span:')
    for example in test_set_features:
        if example.ans_type == 0 and (len(example.start_position) == 0 or len(example.end_position) == 0):
            print(example.qas_id)

    print('long answer:')
    for example in test_set_features:
        if example.ans_type == 0 and (len(example.start_position) > 0 and len(example.end_position) > 0 and
                                      example.end_position[0] - example.start_position[0] > 50):
            print('{}: [{}, {}]'.format(example.qas_id, example.start_position[0], example.end_position[0]))

    print('Test Start.')

    (val_span_loss, val_answer_type_loss, val_support_fact_loss, val_loss, val_answer_type_accu,
     val_span_iou, val_answer_score, val_support_fact_accu, val_support_fact_recall,
     val_support_fact_precision, val_support_fact_f1, val_joint_metric
     ) = RCModel.test(test_set=test_set_features, test_batch_size=32, support_fact_threshold=0.2)

    print('Test Done.')
