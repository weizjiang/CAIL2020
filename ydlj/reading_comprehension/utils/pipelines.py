# !/usr/bin python
# -*- coding:utf-8 -*-
# author:xhades
# datetime:2018/11/04

import os
import threading
import json
import numpy as np
from tflearn.data_utils import pad_sequences
import re
import jieba


def generate_sentence_pairs(sentences, max_context_length=1):

    if max_context_length > 1:
        paragraphs = []
        paragraph = []
        for sentence in sentences:
            if len(sentence) > 0:
                paragraph.append(sentence)
            else:
                if len(paragraph) > 0:
                    paragraphs.append(paragraph)
                paragraph = []

        data_set = []
        for parapraph in paragraphs:
            for idx in range(1, len(parapraph)):
                data_set.append([parapraph[:idx], parapraph[idx]])

    else:
        # data_set = [(sentences[2 * idx], sentences[2 * idx + 1])
        #             for idx in range(int(len(sentences)/2))]
        data_set = [(sentences[idx], sentences[idx + 1])
                    for idx in range(len(sentences) - 1)
                    if len(sentences[idx]) > 0 and len(sentences[idx + 1]) > 0]

    return data_set


def token_to_index(content, vocab):
    result = []
    content_cut = jieba.cut(content)
    for item in content_cut:
        word2id = vocab.get(item)
        if word2id is None:
            if item in [' ', '，']:
                word2id = 0  # <PAD>
            # elif re.match(r'^[a-zA-Z]+$', item):
            #     word2id = 2  # <TOK0>
            elif re.match(r'^1[0-9]{10}$', item):
                word2id = 3  # <TOK1>, mobile phone
            elif re.match(r'^400[0-9]{7}$', item):
                word2id = 4  # <TOK2>, 400 phone
            elif re.match(r'^[0-9]{6,}$', item):
                word2id = 5  # <TOK3>
            elif re.match(r'^[0-9]+\.[0-9]+$', item):
                word2id = 6  # <TOK4>
            elif re.match(r'^[0-9]+:[0-9]+$', item):
                word2id = 7  # <TOK5>
            elif re.match(r'^[a-zA-Z0-9]{10,}$', item):
                word2id = 8  # <TOK6>
            elif re.search(r'[a-zA-Z0-9]', item) is None and len(item)>1:
                # try to separate Chinese words into charactors
                word2id = []
                NoFind = True
                for char in item:
                    char2id = vocab.get(char)
                    if char2id is None:
                        char2id = 1 # <UNK>
                    else:
                        NoFind = False
                    word2id.append(char2id)
                if NoFind:
                    word2id = 1  # <UNK>
            else:
                word2id = 1  # <UNK>

        if type(word2id) is list:
            for ch_id in word2id:
                result.append(ch_id)
        else:
            result.append(word2id)
    return result


def truncate_by_separator(sentence, max_sentence_len):
    """ truncate the sentence to the last separator withing max_sentence_len"""
    if len(sentence) > max_sentence_len:
        sentence = sentence[:max_sentence_len]
        separators = ['，', '；', '：', '、', '。', '？', '！', '…', '—', '“', '”', '（', '）',
                      ',', ';', '!', '?', r'\[', r'\]', r'\{', r'\}', r'\(', r'\)', '<', '>', '\'', '\"', ' ']
        sentence = re.sub(r'([{0}])(?:.(?![{0}]))*$'.format(''.join(separators)), r'\1', sentence)
    return sentence


def separate_sentence(text, separators=None, non_starting_chars=None):
    """ Separate a sentence by any separator listed in separators.
    A separator can be a group of charactors.
    A non-starting charactor following a separator forbids the separation."""
    if separators is None:
        separators = ['。', '？', '！', '；', r'\?', '!', ';']

    if non_starting_chars is None:
        non_starting_chars = ['，', '；', '：', '、', '。', '？', '！', '”', '’', '）', '」', '》', '】', '』', '〉',
                              ',', ';', ':', '!', r'\?', r'\]', r'\}', r'\)']

    text = text.replace('\r', ' ').replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'({0})\s*(?![{1}])'.format('|'.join(separators), ''.join(non_starting_chars)), r'\1\n', text)
    text = re.sub(r'(?<=\n)([{0}])*\n'.format(''.join(non_starting_chars)), '', text)
    if not text.endswith('\n'):
        text = text + '\n'
    return text


def data_word2vec(input_file, num_labels, word2vec_model, labels):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data(includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        num_labels: The number of classes
        word2vec_model: The word2vec model file
    Returns:
        The class Data(includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
#    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.vocab.items()])

    def _create_onehot_labels(labels_index):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.txt'):
        raise IOError("✘ The research data is not a txt file. "
                      "Please preprocess the research data into the txt file.")
    with open(input_file, encoding="utf-8") as fin:
        testid_list = []
        content_index_list = []
        keyword_pos_list = []
        labels_list = []
        onehot_labels_list = []
        labels_bind_list = []
        labels_num_list = []
        total_line = 0
        for eachline in fin:
            data = json.loads(eachline)
            testid = data['testid']
            features_content = data['features_content']
            sample_labels = data['labels']
            labels_index = [labels.index(_) for _ in sample_labels]
            labels_num = data['labels_num']
            testid_list.append(testid)
            content_index_list.append(token_to_index(features_content, vocab))
            labels_list.append(labels_index)
            onehot_labels_list.append(_create_onehot_labels(labels_index))
            labels_num_list.append(labels_num)

            if 'labels_bind' in data.keys():
                labels_bind_list.append(data['labels_bind'])

            if 'feature_words' in data.keys():
                # get the keyword postions in the token sequence
                keywords = set(data['feature_words'])
                keyword_index = []
                if len(keywords) > 0:
                    content_cut = jieba.cut(features_content)
                    cursor = 0
                    token_map = []
                    for item in content_cut:
                        token_map.append([cursor, cursor+len(item)])
                        cursor += len(item)
                    token_map = np.asarray(token_map).T
                    for kw in keywords:
                        for m in re.finditer(kw, features_content):
                            keyword_index = keyword_index + np.where(
                                np.logical_and(token_map[0] >= m.start(), token_map[1] <= m.end()))[0].tolist()
                keyword_pos_list.append(keyword_index)

            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def testid(self):
            return testid_list

        @property
        def tokenindex(self):
            return content_index_list

        @property
        def keywordpos(self):
            return keyword_pos_list

        @property
        def tokenlength(self):
            return [len(l) for l in content_index_list]

        @property
        def labels(self):
            return labels_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

        @property
        def labels_num(self):
            return labels_num_list

        @property
        def labels_bind(self):
            if labels_bind_list:
                return labels_bind_list
            else:
                return None

    return _Data()


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    pad_seq = pad_sequences(data.tokenindex, maxlen=pad_seq_len, value=0.)
    onehot_labels = data.onehot_labels
    actual_length = [np.min([l, pad_seq_len]) for l in data.tokenlength]
    return pad_seq, onehot_labels, actual_length


def map_item2id(items, voc, max_len, none_word=1, lower=False, init_value=0, allow_error=True):
    """
    Convert word to ID.
    Args:
        items: list, 待映射列表
        voc: voc dict
        max_len: int, max length of sentence
        none_word: 未登录词标号,默认为1
        lower: bool, 状态转换为小写
        init_value: default is 0, 初始化的值
        allow_error: default True, 状态允许未登陆词
    Returns:
        arr: np.array, dtype=int32, shape=[max_len,]
    """
    assert type(none_word) == int
    arr = np.zeros((max_len,), dtype='int32') + init_value
    min_range = min(max_len, len(items))
    for i in range(0, min_range):  # 若items长度大于max_len，则被截断
        item = items[i] if not lower else items[i].lower()
        if allow_error:
            arr[i] = voc[item] if item in voc else none_word
        else:
            arr[i] = voc[item]
    return arr


class FilenameIterator(object):
    """ A threadsafe iterator yielding a fixed number of filenames from a given
     folder and looping forever. Can be used for external memory training. """
    
    def __init__(self, dirname, batch_size):
        self.dirname = dirname
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.files = list({filename[:-4] for filename in os.listdir(dirname)})
        self.i = 0
    
    def __iter__(self):
        return self
    
    def next(self):
        with self.lock:
            
            if self.i == len(self.files):
                self.i = 0
            
            batch = self.files[self.i:self.i + self.batch_size]
            if len(batch) < self.batch_size:
                self.i = 0
            else:
                self.i += self.batch_size
            
            return batch
