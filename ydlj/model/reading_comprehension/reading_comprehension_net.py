# !/usr/bin python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
from gensim.models import KeyedVectors
import yaml
import re
import json

from reading_comprehension.utils.data_helpers import get_label_using_scores_by_threshold, cal_metric_batch, cal_F
from reading_comprehension.utils.convert_answer import convert_to_tokens

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../libs'))
from bert import modeling as bert_modeling
from bert import tokenization


class ReadingComprehensionModel:
    def __init__(self, config=None, model_path=None, model_selection='L'):

        if model_path is None:
            if type(config) is not dict:
                if config is None:
                    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'reading_comprehension_config.yml')
                elif type(config) is str and os.path.isfile(config):
                    config_file = config
                else:
                    raise ValueError('Unsupported config: {}'.format(config))
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.load(f.read())

            self.initialize_config(config)
            self.LoadedModel = None

        elif os.path.isdir(model_path):
            self.load_model(model_path, model_selection)
        else:
            raise ValueError("model_path doesn't exist: %s" % model_path)

        # number of answer types, 0: span, 1: yes, 2: no, 3: unknown
        self.num_answer_type = 4

    def initialize_config(self, config={}):
        self.max_input_len = config.get('max_input_len', 512)  # the default max input length for padding
        self.max_num_sentence = config.get('max_num_sentence', 50)
        self.max_num_entity = config.get('max_num_entity', 100)
        self.max_answer_len = config.get('max_answer_len', 50)
        self.restrict_answer_span = config.get('restrict_answer_span', False)
        self.support_fact_threshold = config.get('support_fact_threshold', 0.5)

        self.word_embed_type = config.get('word_embedding_type', 'pretrained')  # word embedding type
        self.word_embed_size = config.get('word_embedding_size', 200)  # dim of embedding
        self.word_embed_trainable = config.get('word_embedding_trainable', False)
        if self.word_embed_type == 'pretrained':
            if os.path.isfile(config.get('word_embedding_file', '')):
                wv_model = KeyedVectors.load(config['word_embedding_file'])
                self.vocab = dict([(k, v.index) for k, v in wv_model.vocab.items()])
                self.vocab_size = len(self.vocab)
                self.init_embeddings = np.zeros([self.vocab_size, config['word_embedding_size']], dtype="float32")
                for key, value in self.vocab.items():
                    if key is not None:
                        self.init_embeddings[value] = wv_model[key]
            else:
                # when loading model, if the file path doesn't exist, the initial word_embedding_file is not over-written.
                pass

        elif self.word_embed_type == 'bert':
            # BERT configurations
            self.bert = config.get('BERT', {})
            self.bert_config = bert_modeling.BertConfig.from_dict(self.bert['bert_config'])
            self.vocab_size = self.bert_config.vocab_size
            self.init_embeddings = None

            if os.path.isfile(self.bert.get('vocab_file', '')):
                vocab_file = self.bert['vocab_file']
            else:
                vocab_file = os.path.join(os.path.dirname(__file__), 'configs', 'vocab.txt')

            self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

            self.bert_frozen_layers = []
            for item in self.bert.get('frozen_layers', []):
                if item == 'embedding':
                    self.bert_frozen_layers.append('bert/embeddings/')
                elif item.isdigit() and int(item) < self.bert_config.num_hidden_layers:
                    self.bert_frozen_layers.append('bert/encoder/layer_%s/' % item)
                else:
                    m = re.match(r'^(\d+)\-(\d+)$', item)
                    if m:
                        start = int(m.group(1))
                        end = min(int(m.group(2)) + 1, self.bert_config.num_hidden_layers)
                        for layer in range(start, end):
                            self.bert_frozen_layers.append('bert/encoder/layer_%d/' % layer)

        self.answer_type_embedding_type = config.get('answer_type_embedding_type', 'FirstPoolDense')
        self.answer_type_embed_size = config.get('answer_type_embed_size', 200)
        self.answer_type_use_query_embedding_only = config.get('answer_type_use_query_embedding_only', False)
        self.answer_pred_use_support_fact_embedding = config.get('answer_pred_use_support_fact_embedding', 'Sentence')
        self.answer_span_predict_model = config.get('answer_span_predict_model', 'None')
        self.answer_span_transformer = config.get('answer_span_transformer', {})
        self.answer_span_self_attention = config.get('answer_span_self_attention', {})

        self.sentence_embedding_type = config.get('sentence_embedding_type', 'MaxPoolDense')
        self.sentence_embed_size = config.get('sentence_embed_size', 200)
        self.sentence_embedding_shared_by_scores = config.get('sentence_embedding_shared_by_scores', True)

        if self.sentence_embedding_type == 'CNN':
            CnnConfig = config.get('CNN', {})
            self.CCNN_NumLayer = CnnConfig.get('NumLayer', 1)
            self.CCNN_FilterSize = CnnConfig.get('FilterSize', [4])
            self.CCNN_ChannelSize = CnnConfig.get('ChannelSize', [256])
            self.CCNN_ConvStride = CnnConfig.get('ConvStride', [1])
            self.CCNN_PoolSize = CnnConfig.get('PoolSize', [16])
            self.CCNN_FcActivation = CnnConfig.get('FcActivation', 'none')
        elif self.sentence_embedding_type == 'BiLSTM':
            BiLSTM_Config = config.get('BiLSTM', {})
            self.lstm_hidden_size = BiLSTM_Config.get('hidden_size', 1024)
            self.lstm_num_layer = BiLSTM_Config.get('num_layer', 1)
            self.lstm_attention_enable = BiLSTM_Config.get('attention_enable', True)

        self.support_fact_reasoning_model = config.get('support_fact_reasoning_model', 'None')
        self.support_fact_transformer = config.get('support_fact_transformer', {})
        self.support_fact_self_attention = config.get('support_fact_self_attention', {})
        self.sentence_entity_connect_type = config.get('sentence_entity_connect_type', 'Tree')

        self.support_fact_loss_all_samples = config.get('support_fact_loss_all_samples', True)
        self.support_fact_loss_type = config.get('support_fact_loss_type', 'Mean')
        self.span_loss_samples = config.get('span_loss_samples', 'NonSpan3Pos')
        self.span_loss_weight = config.get('span_loss_weight', 1.0)
        self.answer_type_loss_weight = config.get('answer_type_loss_weight', 1.0)
        self.support_fact_loss_weight = config.get('support_fact_loss_weight', 1.0)

        self.optimizer = config.get('optimizer', 'Adam')
        self.lr_base = config.get('learning_rate_base', 0.0001)
        self.lr_decay_steps = config.get('learning_rate_decay_steps', 1000)
        self.lr_decay_rate = config.get('learning_rate_decay_rate', 0.95)
        self.lr_min = config.get('learning_rate_min', 0.00001)
        self.momentum_rate = config.get('momentum_rate', 0.9)
        self.momentum_rate_2nd = config.get('momentum_rate_2nd', 0.999)
        self.grad_threshold = config.get('grad_threshold', 5.0)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.L2_REG_LAMBDA = config.get('L2_REG_LAMBDA', 0.0001)
        self.L2_Normalize = config.get('L2_Normalize', False)

        self.batch_size = config.get('batch_size', 200)
        self.save_period = config.get('save_period', 50)
        self.validate_period = config.get('validate_period', 20)
        self.validate_size = config.get('validate_size', 200)


    def episode_iter(self, data_set, episode_size=None, shuffle=True, max_input_len=None, keep_invalid=True):
        """Generate batch vec_data. """
        num_sample = len(data_set)
        if episode_size is None:
            episode_size = self.batch_size
        elif episode_size == -1:
            episode_size = num_sample

        # separate the dataset by single-sentence and multi-sentence context, and make sure each batch only contain one
        # of these two types, in order to avoid too much memory consumption when doing sentence embedding processing.
        num_sentences = np.array([sum([1 for span in item.sent_spans if len(span) == 2]) for item in data_set])

        # subset for single sentence data
        data_set_1_indices = np.where(num_sentences == 1)[0]
        if shuffle:
            np.random.shuffle(data_set_1_indices)
        num_batch_1 = int((len(data_set_1_indices) - 1) / episode_size) + 1
        data_set_1_indices = np.hstack([data_set_1_indices,
                                        -np.ones(num_batch_1*episode_size - len(data_set_1_indices), dtype=np.int32)])
        data_set_1_indices = np.reshape(data_set_1_indices, (num_batch_1, episode_size))

        # subset for multiple sentence data
        data_set_2_indices = np.where(num_sentences > 1)[0]
        if shuffle:
            np.random.shuffle(data_set_2_indices)
        num_batch_2 = int((len(data_set_2_indices) - 1) / episode_size) + 1
        data_set_2_indices = np.hstack([data_set_2_indices,
                                        -np.ones(num_batch_2*episode_size - len(data_set_2_indices), dtype=np.int32)])
        data_set_2_indices = np.reshape(data_set_2_indices, (num_batch_2, episode_size))

        data_set_indices = np.vstack([data_set_1_indices, data_set_2_indices])

        num_episode = num_batch_1 + num_batch_2

        if shuffle:
            batch_indices = np.random.permutation(num_episode)
        else:
            batch_indices = np.arange(num_episode)

        # 0 corresponds to [CLS] token for BERT tokenizer
        IGNORE_INDEX = 0
        for i in range(num_episode):
            cur_batch = [data_set[idx] for idx in data_set_indices[batch_indices[i]] if idx >= 0]
            cur_bsz = len(cur_batch)

            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0

            context_idxs = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            context_mask = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            segment_idxs = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)

            query_mapping = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            # start postions for context senteces and all entities
            start_mapping = np.zeros((cur_bsz, self.max_num_sentence + self.max_num_entity, self.max_input_len),
                                     dtype=np.int32)
            # postions for context senteces and all entities
            all_mapping = np.zeros((cur_bsz, self.max_input_len, self.max_num_sentence + self.max_num_entity),
                                   dtype=np.int32)

            entity_start_mapping = np.zeros((cur_bsz, self.max_num_entity, self.max_input_len), dtype=np.int32)
            entity_mapping = np.zeros((cur_bsz, self.max_input_len, self.max_num_entity), dtype=np.int32)
            # connection between sentences (including query sentence) and entities
            # the first index for dimension 1 corresponds to query sentence
            sent_entity_mapping = np.zeros((cur_bsz, 1 + self.max_num_sentence, self.max_num_entity), dtype=np.int32)

            # Label tensor
            y1 = np.zeros(cur_bsz, dtype=np.int32)
            y2 = np.zeros(cur_bsz, dtype=np.int32)
            q_type = np.zeros(cur_bsz, dtype=np.int32)
            is_support = np.zeros((cur_bsz, self.max_num_sentence), dtype=np.int32)

            valid_sample_idx = []
            for sample_idx in range(len(cur_batch)):
                case = cur_batch[sample_idx]
                is_valid = True

                context_idxs[sample_idx] = case.doc_input_ids
                context_mask[sample_idx] = case.doc_input_mask
                segment_idxs[sample_idx] = case.doc_segment_ids

                # separate sentences and entities
                sent_spans = [span for span in case.sent_spans if len(span) == 2]
                entity_spans = [span for span in case.sent_spans if len(span) == 3]

                for j in range(sent_spans[0][0] - 1):
                    # including [CLS], excluding [SEP]
                    query_mapping[sample_idx, j] = 1

                if case.ans_type == 0:
                    if len(case.start_position) == 0 or len(case.end_position) == 0:
                        y1[sample_idx] = y2[sample_idx] = 0
                        # set to invalid if span is missing
                        is_valid = False
                    elif case.end_position[0] < self.max_input_len:
                        y1[sample_idx] = case.start_position[0]
                        y2[sample_idx] = case.end_position[0]
                    elif case.start_position[0] >= self.max_input_len:
                        # change to 'unknown' in case answer span is out of range
                        q_type[sample_idx] = 3
                        y1[sample_idx] = IGNORE_INDEX
                        y2[sample_idx] = IGNORE_INDEX
                    else:
                        y1[sample_idx] = y2[sample_idx] = 0
                        # set to invalid if span is only partially kept
                        is_valid = False
                    q_type[sample_idx] = 0
                elif case.ans_type == 1:
                    if self.span_loss_samples == 'NonSpan3Pos':
                        # the position of the first [SEP] token
                        y1[sample_idx] = sent_spans[0][0] - 1
                        y2[sample_idx] = sent_spans[0][0] - 1
                    else:
                        y1[sample_idx] = IGNORE_INDEX
                        y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 1
                elif case.ans_type == 2:
                    if self.span_loss_samples == 'NonSpan3Pos':
                        # the position of the second [SEP] token
                        y1[sample_idx] = len(case.doc_tokens) - 1
                        y2[sample_idx] = len(case.doc_tokens) - 1
                    else:
                        y1[sample_idx] = IGNORE_INDEX
                        y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 2
                elif case.ans_type == 3:
                    y1[sample_idx] = IGNORE_INDEX
                    y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 3

                for j, sent_span in enumerate(sent_spans[:self.max_num_sentence]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start < end:
                        if q_type[sample_idx] != 3:
                            # set support fact flag only for non-unknown types
                            is_support[sample_idx, j] = int(is_sp_flag)
                        all_mapping[sample_idx, start:end + 1, j] = 1
                        start_mapping[sample_idx, j, start] = 1

                # # all sentences are connected, including query sentence
                # sent_entity_mapping[sample_idx, :j+2, :j+2] = 1

                # entity mapping
                num_entity = 0
                for entity_idx, entity_span in enumerate(entity_spans[:self.max_num_entity]):
                    sentence_id, start, end = entity_span
                    if sentence_id <= j and start <= end:
                        entity_mapping[sample_idx, start:end + 1, entity_idx] = 1
                        entity_start_mapping[sample_idx, entity_idx, start] = 1
                        sent_entity_mapping[sample_idx, sentence_id + 1, entity_idx] = 1
                        num_entity += 1

                if q_type[sample_idx] != 3 and np.sum(is_support[sample_idx]) == 0:
                    # set to invalid if support fact is missing
                    is_valid = False

                if is_valid or keep_invalid:
                    ids.append(case.qas_id)
                    max_sent_cnt = max(max_sent_cnt, j+1)
                    max_entity_cnt = max(max_entity_cnt, num_entity)
                    valid_sample_idx.append(sample_idx)

            if len(valid_sample_idx) == 0:
                continue

            valid_idx = np.array(valid_sample_idx)
            input_lengths = np.sum(context_mask[valid_idx] > 0.5, axis=1)
            max_c_len = int(input_lengths.max())

            yield {
                'context_idxs': context_idxs[valid_idx, :max_c_len],
                'context_mask': context_mask[valid_idx, :max_c_len],
                'segment_idxs': segment_idxs[valid_idx, :max_c_len],
                'query_mapping': query_mapping[valid_idx, :max_c_len],
                'y1': y1[valid_idx],
                'y2': y2[valid_idx],
                'ids': ids,
                'q_type': q_type[valid_idx],
                'start_mapping': np.concatenate([start_mapping[valid_idx, :max_sent_cnt, :max_c_len],
                                                 entity_start_mapping[valid_idx, :max_entity_cnt, :max_c_len]],
                                                axis=1),
                'all_mapping': np.concatenate([all_mapping[valid_idx, :max_c_len, :max_sent_cnt],
                                               entity_mapping[valid_idx, :max_c_len, :max_entity_cnt]],
                                              axis=2),
                'sent_entity_mapping': sent_entity_mapping[valid_idx, :max_sent_cnt+1, :max_entity_cnt],
                'is_support': is_support[valid_idx, :max_sent_cnt],
            }

    def predict_answer_span(self, input_tokens, span_start_prob, span_end_prob):
        batch_size = input_tokens.shape[0]
        input_len = input_tokens.shape[1]
        span_prob = np.expand_dims(span_start_prob, axis=2) * np.expand_dims(span_end_prob, axis=1)
        span_mask = np.tril(np.triu(np.ones((input_len, input_len)), 0), self.max_answer_len)
        
        # generate mask considerting the restriction characters, which is not allowed in the answer span
        # note that the 'end' pos is included in the answer span
        restrict_chars = ['，', '。', '；', '！', '？']
        restrict_tokens = [self.bert_tokenizer.vocab[char] for char in restrict_chars]
        restrict_mask = np.zeros([batch_size, input_len, input_len])
        for sample_idx in range(batch_size):
            for start_idx in range(input_len):
                if input_tokens[sample_idx, start_idx] in restrict_tokens:
                    continue
                for end_idx in range(start_idx, input_len):
                    if input_tokens[sample_idx, end_idx] in restrict_tokens:
                        break
                    restrict_mask[sample_idx, start_idx, end_idx] = 1

        span_prob = span_prob * np.expand_dims(span_mask, 0) * restrict_mask

        span_start_pos = np.argmax(np.max(span_prob, axis=2), axis=1)
        span_end_pos = np.argmax(np.max(span_prob, axis=1), axis=1)

        return span_start_pos, span_end_pos

    def predict_answer_type_and_support_fact(self, answer_type_prob, support_fact_prob, support_fact_threshold=None):
        answer_type_predicts = np.argmax(answer_type_prob, axis=1)

        if support_fact_threshold is None:
            support_fact_threshold = self.support_fact_threshold
        support_fact_predicts, _ = get_label_using_scores_by_threshold(
            support_fact_prob,
            threshold=support_fact_threshold,
            topN=-1,
            allow_empty=False
        )

        # 'unkown' type answer should have no supporting fact
        for cnt, answer_type_predict in enumerate(answer_type_predicts):
            if answer_type_predict == 3:
                support_fact_predicts[cnt] = []

        for support_facts in support_fact_predicts:
            support_facts.sort()

        return answer_type_predicts, support_fact_predicts

    def evaluate(self, predict, gold):
        batch_size = gold['context_idxs'].shape[0]

        answer_type_predicts = np.array(predict['answer_type'])
        answer_type_accu = np.sum(answer_type_predicts == gold['q_type']) / batch_size

        avg_answer_score = 0.
        span_iou = []

        for idx in range(batch_size):
            if answer_type_predicts[idx] != gold['q_type'][idx]:
                answer_score = 0.
            elif gold['q_type'][idx] == 0:
                # for span type answer, calculate IoU as the score metric
                intersection_start = max([predict['span_start_pos'][idx], gold['y1'][idx]])
                intersection_end = min([predict['span_end_pos'][idx], gold['y2'][idx]])
                union_start = min([predict['span_start_pos'][idx], gold['y1'][idx]])
                union_end = max([predict['span_end_pos'][idx], gold['y2'][idx]])
                answer_score = max(intersection_end - intersection_start + 1, 0) / (union_end - union_start + 1)
                span_iou.append(answer_score)
            else:
                answer_score = 1.

            avg_answer_score += answer_score

        # the combined score of answer type and span IoU
        avg_answer_score = avg_answer_score / batch_size

        num_span = len(span_iou)
        if num_span > 0:
            avg_span_iou = np.mean(span_iou)
        else:
            avg_span_iou = -1

        # number for questions what need suppfort facts (not 'unknown' type)
        # only consider the neither prediction type or gold type is 'unknown', to avoid 'nan' recall or precision.
        # also consider the 'is_support' indicator in gold, since the support fact sentence may be truncated.
        num_support_fact = np.sum(
            np.logical_and(answer_type_predicts != 3, np.any(gold['is_support'], axis=1))
        )

        support_fact_labels = [list(np.nonzero(gold['is_support'][idx])[0]) for idx in range(batch_size)]

        support_fact_accu, support_fact_recall, support_fact_precision, support_fact_f1 = cal_metric_batch(
            predict['support_fact'], support_fact_labels)

        joint_metric = avg_answer_score * support_fact_f1

        return (answer_type_accu, avg_span_iou, avg_answer_score, support_fact_accu, support_fact_recall,
                support_fact_precision, support_fact_f1, joint_metric, num_span, num_support_fact)

    def load_model(self, model_path, selection='L'):

        if selection == 'B':
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(model_path, 'bestcheckpoints'))
        else:
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(model_path, 'checkpoints'))

        # overwrite the original configuration
        configFile = os.path.join(model_path, 'reading_comprehension_config.yml')
        if os.path.isfile(configFile):
            with open(configFile, 'r', encoding='utf-8') as f:
                config = yaml.load(f.read())
                self.initialize_config(config)

        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()

            # Load the saved meta graph and restore variables
            tf.train.import_meta_graph("{0}.meta".format(checkpoint_file)).restore(self.session, checkpoint_file)

            self.input_sentence = graph.get_tensor_by_name("input_sentence:0")
            self.input_mask = graph.get_tensor_by_name("input_mask:0")
            self.input_answer_type = graph.get_tensor_by_name("input_answer_type:0")
            self.input_span_start = graph.get_tensor_by_name("input_span_start:0")
            self.input_span_end = graph.get_tensor_by_name("input_span_end:0")
            self.input_sentence_mapping = graph.get_tensor_by_name("input_sentence_mapping:0")
            self.input_sent_entity_mapping = graph.get_tensor_by_name("input_sent_entity_mapping:0")
            self.input_support_facts = graph.get_tensor_by_name("input_support_facts:0")
            self.is_training = graph.get_tensor_by_name("is_training:0")

            self.span_start_pos = graph.get_tensor_by_name("span_start_pos:0")
            self.span_end_pos = graph.get_tensor_by_name("span_end_pos:0")
            self.span_start_prob = graph.get_tensor_by_name("span_start_prob:0")
            self.span_end_prob = graph.get_tensor_by_name("span_end_prob:0")
            self.answer_type_prob = graph.get_tensor_by_name("answer_type_prob:0")
            self.support_fact_prob = graph.get_tensor_by_name("support_fact_prob:0")
            self.span_loss = graph.get_tensor_by_name("span_loss:0")
            self.answer_type_loss = graph.get_tensor_by_name("answer_type_loss:0")
            self.support_fact_loss = graph.get_tensor_by_name("support_fact_loss:0")
            self.reg_loss = graph.get_tensor_by_name("reg_loss:0")
            self.loss = graph.get_tensor_by_name("loss:0")
            self.train_op = graph.get_operation_by_name("train_op")
            self.learning_rate = graph.get_tensor_by_name("learning_rate:0")
            global_steps = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
            if len(global_steps) > 0:
                self.global_step = global_steps[0]

        self.LoadedModel = checkpoint_file

    def test(self, test_set, test_batch_size=None, max_input_len=None, support_fact_threshold=None,
             restrict_answer_span=None):

        if restrict_answer_span is None:
            restrict_answer_span = self.restrict_answer_span

        val_span_loss_list = []
        val_answer_type_loss_list = []
        val_support_fact_loss_list = []
        val_loss_list = []
        val_answer_type_accu_list = []
        val_span_iou_list = []
        val_answer_score_list = []
        val_support_fact_accu_list = []
        val_support_fact_recall_list = []
        val_support_fact_precision_list = []
        val_support_fact_f1_list = []
        val_joint_metric_list = []
        val_num_span_list = []
        val_num_support_fact_list = []

        all_answer_type_predicts = np.array([], dtype=np.int32)
        all_answer_type_labels = np.array([], dtype=np.int32)
        for val_batch in self.episode_iter(test_set, test_batch_size, shuffle=False, max_input_len=max_input_len):
            (val_step_span_start_pos, val_step_span_end_pos, val_step_span_start_prob, val_step_span_end_prob,
             val_step_answer_type_prob, val_step_support_fact_prob,
             val_step_span_loss, val_step_answer_type_loss, val_step_support_fact_loss, val_step_loss
             ) = self.session.run([
                self.span_start_pos, self.span_end_pos, self.span_start_prob, self.span_end_prob,
                self.answer_type_prob, self.support_fact_prob, self.span_loss, self.answer_type_loss,
                self.support_fact_loss, self.loss],
                feed_dict={
                    self.input_sentence: val_batch['context_idxs'],
                    self.input_mask: val_batch['context_mask'],
                    self.input_answer_type: val_batch['q_type'],
                    self.input_span_start: val_batch['y1'],
                    self.input_span_end: val_batch['y2'],
                    self.input_sentence_mapping: val_batch['all_mapping'],
                    self.input_sent_entity_mapping: val_batch['sent_entity_mapping'],
                    self.input_support_facts: val_batch['is_support'],
                    self.is_training: False
                }
            )

            if restrict_answer_span:
                span_start_predicts, span_end_predicts = self.predict_answer_span(
                    val_batch['context_idxs'], val_step_span_start_prob, val_step_span_end_prob)
            else:
                span_start_predicts = val_step_span_start_pos
                span_end_predicts = val_step_span_end_pos

            answer_type_predicts, support_fact_predicts = self.predict_answer_type_and_support_fact(
                val_step_answer_type_prob, val_step_support_fact_prob, support_fact_threshold)

            all_answer_type_predicts = np.hstack([all_answer_type_predicts, answer_type_predicts])
            all_answer_type_labels = np.hstack([all_answer_type_labels, val_batch['q_type']])

            (val_step_answer_type_accu,
             val_step_span_iou,
             val_step_answer_score,
             val_step_support_fact_accu,
             val_step_support_fact_recall,
             val_step_support_fact_precision,
             val_step_support_fact_f1,
             val_step_joint_metric,
             val_step_num_span,
             val_step_num_support_fact
             ) = self.evaluate(
                {
                    'span_start_pos': span_start_predicts,
                    'span_end_pos': span_end_predicts,
                    'answer_type': answer_type_predicts,
                    'support_fact': support_fact_predicts
                },
                val_batch
            )

            val_span_loss_list.append(val_step_span_loss)
            val_answer_type_loss_list.append(val_step_answer_type_loss)
            val_support_fact_loss_list.append(val_step_support_fact_loss)
            val_loss_list.append(val_step_loss)
            val_answer_type_accu_list.append(val_step_answer_type_accu)
            val_span_iou_list.append(val_step_span_iou)
            val_answer_score_list.append(val_step_answer_score)
            val_support_fact_accu_list.append(val_step_support_fact_accu)
            val_support_fact_recall_list.append(val_step_support_fact_recall)
            val_support_fact_precision_list.append(val_step_support_fact_precision)
            val_support_fact_f1_list.append(val_step_support_fact_f1)
            val_joint_metric_list.append(val_step_joint_metric)
            val_num_span_list.append(val_step_num_span)
            val_num_support_fact_list.append(val_step_num_support_fact)

        val_num_span_ary = np.array(val_num_span_list)
        span_idx = val_num_span_ary > 0
        val_span_loss_ary = np.array(val_span_loss_list)
        val_span_loss = np.sum(val_span_loss_ary[span_idx] * val_num_span_ary[span_idx]) / np.sum(val_num_span_ary)

        val_answer_type_loss = np.mean(val_answer_type_loss_list)
        val_support_fact_loss = np.mean(val_support_fact_loss_list)
        val_loss = np.mean(val_loss_list)
        val_answer_type_accu = np.mean(val_answer_type_accu_list)
        val_span_iou_ary = np.array(val_span_iou_list)
        val_span_iou = np.sum(val_span_iou_ary[span_idx] * val_num_span_ary[span_idx]) / np.sum(val_num_span_ary)
        val_answer_score = np.mean(val_answer_score_list)

        val_num_support_fact_ary = np.array(val_num_support_fact_list)
        support_fact_idx = val_num_support_fact_ary > 0
        val_support_fact_accu_ary = np.array(val_support_fact_accu_list)
        val_support_fact_accu = np.sum(val_support_fact_accu_ary[support_fact_idx] *
                                       val_num_support_fact_ary[support_fact_idx]) / np.sum(val_num_support_fact_ary)
        val_support_fact_recall_ary = np.array(val_support_fact_recall_list)
        val_support_fact_recall = np.sum(
            val_support_fact_recall_ary[support_fact_idx] * val_num_support_fact_ary[support_fact_idx]
        ) / np.sum(val_num_support_fact_ary)
        val_support_fact_precision_ary = np.array(val_support_fact_precision_list)
        val_support_fact_precision = np.sum(
            val_support_fact_precision_ary[support_fact_idx] * val_num_support_fact_ary[support_fact_idx]
        ) / np.sum(val_num_support_fact_ary)
        val_support_fact_f1_ary = np.array(val_support_fact_f1_list)
        val_support_fact_f1 = np.sum(
            val_support_fact_f1_ary[support_fact_idx] * val_num_support_fact_ary[support_fact_idx]
        ) / np.sum(val_num_support_fact_ary)
        val_joint_metric_ary = np.array(val_joint_metric_list)
        val_joint_metric = np.sum(
            val_joint_metric_ary[support_fact_idx] * val_num_support_fact_ary[support_fact_idx]
        ) / np.sum(val_num_support_fact_ary)

        print("===== answer type results =====")
        answer_types = ['span', 'yes', 'no', 'unknown']
        for label_idx in range(self.num_answer_type):
            if sum(all_answer_type_predicts == label_idx) == 0:
                precision = float('NaN')
            else:
                precision = sum(np.logical_and(all_answer_type_predicts == label_idx,
                                               all_answer_type_labels == all_answer_type_predicts)) / \
                            sum(all_answer_type_predicts == label_idx)
            if sum(all_answer_type_labels == label_idx) == 0:
                recall = float('NaN')
            else:
                recall = sum(np.logical_and(all_answer_type_labels == label_idx,
                                            all_answer_type_labels == all_answer_type_predicts)) / \
                         sum(all_answer_type_labels == label_idx)
            if np.isnan(recall) or np.isnan(precision):
                f1 = float('NaN')
            else:
                f1 = cal_F(recall, precision)
            print("{:s}: Recall {:.3f}, Precision {:.3f}, F1 {:.3f}"
                  .format(answer_types[label_idx], recall, precision, f1))

        print(" test ".center(25, "="))
        print("span_loss:{:.4f}\tanswer_type_loss:{:.4f}\tsupport_fact_loss:{:.4f}\tloss:{:.4f}\n"
              "answer_type_accu:{:.4f}\tspan_iou:{:.4f}\tanswer_score:{:.4f}\t"
              "support_fact_accu:{:.4f}\tsupport_fact_recall:{:.4f}\tsupport_fact_precision:{:.4f}\t"
              "support_fact_f1:{:.4f}\tjoint_metric:{:.4f}".format(
                val_span_loss, val_answer_type_loss, val_support_fact_loss, val_loss, val_answer_type_accu,
                val_span_iou, val_answer_score, val_support_fact_accu, val_support_fact_recall,
                val_support_fact_precision, val_support_fact_f1, val_joint_metric))

        return (val_span_loss, val_answer_type_loss, val_support_fact_loss, val_loss, val_answer_type_accu,
                val_span_iou, val_answer_score, val_support_fact_accu, val_support_fact_recall,
                val_support_fact_precision, val_support_fact_f1, val_joint_metric)

    def predict(self, examples, features, batch_size=None, max_input_len=None, test_per_sample=True,
                support_fact_threshold=None, result_file=None, restrict_answer_span=None):
        num_sample = len(examples)
        if batch_size is None:
            batch_size = self.batch_size
        elif batch_size == -1:
            batch_size = num_sample

        if support_fact_threshold is None:
            support_fact_threshold = self.support_fact_threshold

        if restrict_answer_span is None:
            restrict_answer_span = self.restrict_answer_span

        if result_file is None:
            restrict_str = '_restrict_span' if restrict_answer_span else ''
            m = re.search(r'[\\/](rc_[\d-]*)[\\/]', self.LoadedModel)
            if m:
                model_name = m.group(1)
                result_file = 'results/result_{}_threshold_{}{}.json'.format(
                    model_name, support_fact_threshold, restrict_str)
            else:
                result_file = 'results/result_threshold_{}{}.json'.format(support_fact_threshold, restrict_str)

        example_dict = {e.qas_id: e for e in examples}
        feature_dict = {f.qas_id: f for f in features}

        answer_dict = {}
        support_fact_dict = {}

        for val_batch in self.episode_iter(features, batch_size, shuffle=False, max_input_len=max_input_len):
            (span_start_pos, span_end_pos, span_start_prob, span_end_prob, answer_type_prob, support_fact_prob
             ) = self.session.run([
                self.span_start_pos, self.span_end_pos, self.span_start_prob, self.span_end_prob, self.answer_type_prob,
                self.support_fact_prob],
                feed_dict={
                    self.input_sentence: val_batch['context_idxs'],
                    self.input_mask: val_batch['context_mask'],
                    self.input_sentence_mapping: val_batch['all_mapping'],
                    self.input_sent_entity_mapping: val_batch['sent_entity_mapping'],
                    self.is_training: False
                }
            )

            if restrict_answer_span:
                span_start_predicts, span_end_predicts = self.predict_answer_span(val_batch['context_idxs'],
                                                                                  span_start_prob, span_end_prob)
            else:
                span_start_predicts = span_start_pos
                span_end_predicts = span_end_pos

            answer_type_predicts, support_fact_predicts = self.predict_answer_type_and_support_fact(
                answer_type_prob, support_fact_prob, support_fact_threshold)

            ids = val_batch['ids']
            answer_dict_batch = convert_to_tokens(example_dict, feature_dict, ids,
                                                  list(span_start_predicts),
                                                  list(span_end_predicts),
                                                  list(answer_type_predicts),
                                                  self.bert_tokenizer)

            answer_dict_batch = {key: value.replace(" ", "") for key, value in answer_dict_batch.items()}

            support_fact_dict_batch = {}
            for sample_idx, support_facts in enumerate(support_fact_predicts):
                cur_id = ids[sample_idx]
                support_fact_dict_batch[cur_id] = [
                    example_dict[cur_id].sent_names[sentence_idx] for sentence_idx in support_facts]

            for cur_id in ids:
                orig_id = cur_id
                # convert to original case id and sentence ids for examples with shifted context
                sentence_id_shift = 0
                if type(cur_id) is str:
                    m = re.search(r'^(.+)_SHIFT(\d+)$', orig_id)
                    if m:
                        orig_id = m.group(1)
                        sentence_id_shift = int(m.group(2))
                    m = re.search(r'^INT\((\d+)\)$', orig_id)
                    if m:
                        orig_id = int(m.group(1))

                if orig_id not in answer_dict or answer_dict[orig_id] == 'unknown':
                    # over-write previously predicted answer only if it's 'unknown'
                    answer_dict[orig_id] = answer_dict_batch[cur_id]
                    if sentence_id_shift > 0 and len(support_fact_dict_batch[cur_id]) > 0:
                        # shift the sentence ids. This only supports single paragrpah context.
                        support_fact_dict[orig_id] = [
                            (sent[0], sent[1] + sentence_id_shift) for sent in support_fact_dict_batch[cur_id]]
                    else:
                        support_fact_dict[orig_id] = support_fact_dict_batch[cur_id]

        if test_per_sample:
            for sample_idx, cur_id in enumerate(ids):
                print('----------- {}'.format(cur_id))
                answer_correct = answer_dict[cur_id] == example_dict[cur_id].orig_answer_text
                print('context: {}'.format(''.join(example_dict[cur_id].doc_tokens)))
                print('question: {}'.format(example_dict[cur_id].question_text))
                print('predict answer: {} {}'.format(answer_dict[cur_id], '✔' if answer_correct else '✘'))
                print("gold answer: {}".format(example_dict[cur_id].orig_answer_text))
                print('predict support facts: {}'.format([item[1] for item in support_fact_dict[cur_id]]))
                print("gold support facts: {}".format(example_dict[cur_id].sup_fact_id))

        prediction = {'answer': answer_dict, 'sp': support_fact_dict}
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w', encoding='utf8') as f:
            json.dump(prediction, f, indent=4, ensure_ascii=False)
