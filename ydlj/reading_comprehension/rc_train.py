# !/usr/bin python
# -*- coding:utf-8 -*-

import yaml
import os
import sys
import argparse
import json
import gzip
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.reading_comprehension_net import ReadingComprehensionModel
from reading_comprehension.data_process import InputFeatures, Example

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='configuration file path', default='configs/reading_comprehension_config.yml')

    return parser.parse_args(argv)


def open_dataset_file(file_name):
    if file_name.endswith('.gz'):
        return gzip.open(file_name, 'rb')
    elif file_name.endswith('.pkl'):
        return open(file_name, 'rb')
    elif file_name.endswith('.json') or file_name.endswith('.txt'):
        return open(file_name, 'r', encoding='utf-8')
    else:
        raise NotImplementedError


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])

    if os.path.isfile(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            doc = f.read()
            config = yaml.load(doc)
    else:
        print("ERROR: config file not found.")

    RCModel = ReadingComprehensionModel(config)

    DataSet = config.pop('DataSet')
    with open_dataset_file(DataSet.get('TrainSetExampleFile')) as f:
        train_set_examples = pickle.load(f)
    with open_dataset_file(DataSet.get('TrainSetFeatureFile')) as f:
        train_set_features = pickle.load(f)
    with open_dataset_file(DataSet.get('DevSetExampleFile')) as f:
        dev_set_examples = pickle.load(f)
    with open_dataset_file(DataSet.get('DevSetFeatureFile')) as f:
        dev_set_features = pickle.load(f)

    # # continue training
    # RCModel.load_model(r'runs/rc_20200715-172519', 'L')

    RCModel.fit(train_set=train_set_features,
                validation_set=dev_set_features,
                epochs=config['num_epoch']
                )

    # RCModel.test(test_set=TestSet, test_batch_size=None)
