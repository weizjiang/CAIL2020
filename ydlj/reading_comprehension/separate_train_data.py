# !/usr/bin python
# -*- coding:utf-8 -*-
import os
import sys
import gzip
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from reading_comprehension.data_process import InputFeatures, Example


def save_to_folder(features, feature_folder, examples=None, example_folder=None):
    os.makedirs(feature_folder, exist_ok=True)
    if example_folder is not None:
        os.makedirs(example_folder, exist_ok=True)

    data_length = len(features)
    if examples is not None:
        assert len(examples) == data_length

    permutation = np.random.permutation(data_length)

    num_sample_per_subset = 1000
    num_subset = int(np.ceil(data_length / num_sample_per_subset))
    num_digit = int(np.ceil(np.log10(num_subset)))
    for subset_idx in range(num_subset):
        subset_start = subset_idx * num_sample_per_subset
        subset_end = min((subset_idx + 1) * num_sample_per_subset, data_length)
        feature_subset = [features[idx] for idx in permutation[subset_start:subset_end]]

        subset_file_name = ('{:0>%dd}.pkl.gz' % num_digit).format(subset_idx)
        feature_subset_file = os.path.join(feature_folder, subset_file_name)
        with gzip.open(feature_subset_file, 'wb') as fout:
            pickle.dump(feature_subset, fout)

        if examples is not None and example_folder is not None:
            example_subset = [examples[idx] for idx in permutation[subset_start:subset_end]]
            example_subset_file = os.path.join(example_folder, subset_file_name)
            with gzip.open(example_subset_file, 'wb') as fout:
                pickle.dump(example_subset, fout)



if __name__ == '__main__':
    data_set_example_file = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_example.pkl.gz'
    data_set_feature_file = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_feature.pkl.gz'

    data_set_example_folder = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_examples/'
    data_set_feature_folder = '../data/data_big_combine2019all_cmrc2018all_1sentence_augmented/train_features/'

    with gzip.open(data_set_example_file, 'rb') as f:
        examples = pickle.load(f)

    with gzip.open(data_set_feature_file, 'rb') as f:
        features = pickle.load(f)

    save_to_folder(features, data_set_feature_folder, examples, data_set_example_folder)
