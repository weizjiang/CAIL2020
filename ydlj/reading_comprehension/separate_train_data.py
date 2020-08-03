# !/usr/bin python
# -*- coding:utf-8 -*-
import os
import sys
import gzip
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.data_process import InputFeatures, Example


if __name__ == '__main__':
    data_set_example_file = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_example.pkl.gz'
    data_set_feature_file = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_feature.pkl.gz'

    data_set_example_folder = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_examples/'
    data_set_feature_folder = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_features/'
    os.makedirs(data_set_example_folder, exist_ok=True)
    os.makedirs(data_set_feature_folder, exist_ok=True)

    with gzip.open(data_set_example_file, 'rb') as f:
        examples = pickle.load(f)

    with gzip.open(data_set_feature_file, 'rb') as f:
        features = pickle.load(f)

    data_length = len(examples)
    assert len(features) == data_length

    permutation = np.random.permutation(data_length)

    num_sample_per_subset = 1000
    num_subset = int(np.ceil(data_length / num_sample_per_subset))
    num_digit = int(np.ceil(np.log10(num_subset)))
    for subset_idx in range(num_subset):
        subset_start = subset_idx*num_sample_per_subset
        subset_end = min((subset_idx+1)*num_sample_per_subset, data_length)
        example_subset = [examples[idx] for idx in permutation[subset_start:subset_end]]
        feature_subset = [features[idx] for idx in permutation[subset_start:subset_end]]

        subset_file_name = ('{:0>%dd}.pkl.gz' % num_digit).format(subset_idx)
        example_subset_file = os.path.join(data_set_example_folder, subset_file_name)
        with gzip.open(example_subset_file, 'wb') as fout:
            pickle.dump(example_subset, fout)

        feature_subset_file = os.path.join(data_set_feature_folder, subset_file_name)
        with gzip.open(feature_subset_file, 'wb') as fout:
            pickle.dump(feature_subset, fout)

