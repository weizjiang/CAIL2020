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

    test_set_example_file = '../data/data_correct_span/dev_example.pkl.gz'
    test_set_feature_file = '../data/data_correct_span/dev_feature.pkl.gz'
    with gzip.open(test_set_example_file, 'rb') as f:
        test_set_examples = pickle.load(f)
    with gzip.open(test_set_feature_file, 'rb') as f:
        test_set_features = pickle.load(f)

    # start = timer()

    RCModel.predict(examples=test_set_examples, features=test_set_features, batch_size=32, test_per_sample=False, 
                    support_fact_threshold=0.2, restrict_answer_span=True)

    # end = timer()
    # print('- Done in %.3f seconds -' % (end - start))

