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
    # RCModel.load_model(r'runs/rc_20200715-172519', 'B')
    # RCModel.load_model(r'runs/rc_20200716-110212', 'L')
    # RCModel.load_model(r'runs/rc_20200716-102909', 'L')
    # RCModel.load_model(r'runs/rc_20200716-182107', 'B')
    # RCModel.load_model(r'runs/rc_20200717-165251', 'B')
    
    # test_set_example_file = '../data/data_span_within_support_facts/dev_example.pkl.gz'
    # test_set_feature_file = '../data/data_span_within_support_facts/dev_feature.pkl.gz'
    
    # RCModel.load_model(r'runs/rc_20200720-204118', 'B')
    # RCModel.load_model(r'runs/rc_20200720-205600', 'B')
    # RCModel.load_model(r'runs/rc_20200721-095103', 'B')
    # RCModel.load_model(r'runs/rc_20200721-104351', 'B')
    # RCModel.load_model(r'runs/rc_20200722-122420', 'B')
    # RCModel.load_model(r'runs/rc_20200722-152557', 'B')
    # RCModel.load_model(r'runs/rc_20200723-135847', 'B')
    # RCModel.load_model(r'runs/rc_20200723-211344', 'B')
    # RCModel.load_model(r'runs/rc_20200726-185728', 'B')
    # RCModel.load_model(r'runs/rc_20200727-110232', 'B')
    # RCModel.load_model(r'runs/rc_20200727-112404', 'B')
    
    # test_set_example_file = '../data/data_big_combine2019all_cmrc2018all_1sentence/dev_example_noEntity.pkl.gz'
    # test_set_feature_file = '../data/data_big_combine2019all_cmrc2018all_1sentence/dev_feature_noEntity.pkl.gz'
    
    # RCModel.load_model(r'runs/rc_20200727-204614', 'B')
    # RCModel.load_model(r'runs/rc_20200727-210204', 'B')
    # RCModel.load_model(r'runs/rc_20200728-160636', 'B')
    # RCModel.load_model(r'runs/rc_20200729-121944', 'B')
    # RCModel.load_model(r'runs/rc_20200729-115544', 'B')
    # RCModel.load_model(r'runs/rc_20200730-210101', 'B')
    RCModel.load_model(r'runs/rc_20200731-101153', 'B')
    # RCModel.load_model(r'runs/rc_20200731-112631', 'B')
    # RCModel.load_model(r'runs/rc_20200731-213550', 'B')
    # RCModel.load_model(r'runs/rc_20200801-122142', 'B')
    # RCModel.load_model(r'runs/rc_20200801-152842', 'B')
    # RCModel.load_model(r'runs/rc_20200802-183021', 'B')
    # RCModel.load_model(r'runs/rc_20200802-184044', 'B')
    # RCModel.load_model(r'runs/rc_20200803-205745', 'B')
    # RCModel.load_model(r'runs/rc_20200804-210301', 'B')
    # RCModel.load_model(r'runs/rc_20200804-211313', 'B')
    # RCModel.load_model(r'runs/rc_20200805-162856', 'B')
    # RCModel.load_model(r'runs/rc_20200805-203754', 'B')
    # RCModel.load_model(r'runs/rc_20200805-204554', 'B')
    # RCModel.load_model(r'runs/rc_20200806-115812', 'B')
    # RCModel.load_model(r'runs/rc_20200806-175529', 'B')
    # RCModel.load_model(r'runs/rc_20200807-142854', 'B')
    # RCModel.load_model(r'runs/rc_20200807-212547', 'B')
    # RCModel.load_model(r'runs/rc_20200808-191119', 'B')
    # RCModel.load_model(r'runs/rc_20200809-212409', 'B')
    # RCModel.load_model(r'runs/rc_20200809-215147', 'B')
    # RCModel.load_model(r'runs/rc_20200811-211742', 'B')
    # RCModel.load_model(r'runs/rc_20200811-212027', 'B')
    # RCModel.load_model(r'runs/rc_20200811-212426', 'B')
    # RCModel.load_model(r'runs/rc_20200812-205655', 'B')
    # RCModel.load_model(r'runs/rc_20200813-203652', 'B')
    # RCModel.load_model(r'runs/rc_20200813-204844', 'B')
    # RCModel.load_model(r'runs/rc_20200814-111301', 'B')
    # RCModel.load_model(r'runs/rc_20200814-120452', 'B')
    # RCModel.load_model(r'runs/rc_20200817-001020', 'B')
    # RCModel.load_model(r'runs/rc_20200817-001204', 'B')
    # RCModel.load_model(r'runs/rc_20200817-001603', 'B')
    # RCModel.load_model(r'runs/rc_20200817-204740', 'B')
    # RCModel.load_model(r'runs/rc_20200817-210827', 'B')
    
    test_set_example_file = '../data/data_big_combine2019all_cmrc2018all_1sentence/dev_example.pkl.gz'
    test_set_feature_file = '../data/data_big_combine2019all_cmrc2018all_1sentence/dev_feature.pkl.gz'
    
    with gzip.open(test_set_example_file, 'rb') as f:
        test_set_examples = pickle.load(f)
    with gzip.open(test_set_feature_file, 'rb') as f:
        test_set_features = pickle.load(f)

    # start = timer()

    RCModel.predict(examples=test_set_examples, features=test_set_features, batch_size=32, test_per_sample=True, 
                    support_fact_threshold=0.5, restrict_answer_span=True)

    # end = timer()
    # print('- Done in %.3f seconds -' % (end - start))

