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

    RCModel.test(test_set=test_set_features, test_batch_size=2)

