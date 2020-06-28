# !/usr/bin python
# -*- coding:utf-8 -*-

import yaml
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.reading_comprehension_net import ReadingComprehensionModel


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='configuration file path', default='configs/reading_comprehension_config.yml')

    return parser.parse_args(argv)


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
    TrainPercentage = DataSet.get('TrainPercentage', '-1')

    with open(DataSet.get('TrainSetFile'), "r", encoding='utf-8') as f:
        TrainSet = json.load(f)
    with open(DataSet.get('TestSetFile'), "r", encoding='utf-8') as f:
        TestSet = json.load(f)

    # # continue training
    # RCModel.load_model(r'runs\qe_20200222-235523', 'L')

    RCModel.fit(train_set=TrainSet,
                validation_set=TestSet,
                epochs=config['num_epoch']
                )

    # RCModel.test(test_set=TestSet, test_batch_size=None, test_per_sample=False)
