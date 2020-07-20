# !/usr/bin python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
import sys
import random
import time
from shutil import copyfile
from gensim.models import KeyedVectors
import yaml
from tflearn.data_utils import pad_sequences
import re
import json

from reading_comprehension.utils import checkmate as cm
from reading_comprehension.utils.pipelines import token_to_index as tokenize
from reading_comprehension.utils.data_helpers import get_label_using_scores_by_threshold, cal_metric_batch, cal_F
from reading_comprehension.utils.convert_answer import convert_to_tokens

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../libs'))
from bert import modeling as bert_modeling
from bert import optimization, tokenization, run_classifier


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
        self.answer_pred_use_support_fact_embedding = config.get('answer_pred_use_support_fact_embedding', False)
        self.answer_span_predict_model = config.get('answer_span_predict_model', 'None')
        self.answer_span_transformer = config.get('answer_span_transformer', {})

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

        self.support_fact_loss_all_samples = config.get('support_fact_loss_all_samples', True)
        self.support_fact_loss_type = config.get('support_fact_loss_type', 'Mean')
        self.span_loss_all_samples = config.get('span_loss_all_samples', False)
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

    def initialize_session(self):
        print("start to initialize the variables ")
        print("create session. ")

        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()

            self.regularizer = tf.contrib.layers.l2_regularizer(self.L2_REG_LAMBDA)

            # shape: batch_size x input_length
            self.input_sentence = tf.placeholder(tf.int32, [None, None], name="input_sentence")
            # shape: batch_size x input_length
            self.input_mask = tf.placeholder(tf.float32, [None, None], name='input_mask')
            # shape: batch_size, type index
            self.input_answer_type = tf.placeholder(tf.int32, shape=[None], name="input_answer_type")
            # shape: batch_size, token index
            self.input_span_start = tf.placeholder(tf.int32, shape=[None], name="input_span_start")
            # shape: batch_size, token index
            self.input_span_end = tf.placeholder(tf.int32, shape=[None], name="input_span_end")
            # shape: batch_size x input_length x num_sentence
            self.input_sentence_mapping = tf.placeholder(tf.float32, shape=[None, None, None],
                                                         name="input_sentence_mapping")
            # shape: batch_size x num_sentence
            self.input_support_facts = tf.placeholder(tf.float32, shape=[None, None], name="input_support_facts")
            # shape: bool scaler
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            self.build_token_embeddings()

            self.build_graph()

            self.define_loss_optimizer()

            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.local_variables_initializer())

        print("initilized finished!!!")

    def build_token_embeddings(self):
        # init embedding matrix
        if self.word_embed_type == 'pretrained' and isinstance(self.init_embeddings, np.ndarray):
            self.embeddings = tf.Variable(self.init_embeddings, name='Embedding', trainable=self.word_embed_trainable)
        elif self.word_embed_type != 'bert':
            self.embeddings = tf.get_variable("Embedding", [self.vocab_size, self.word_embed_size],
                                              initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1,
                                                                                        dtype=tf.float32),
                                              trainable=self.word_embed_trainable)
        else:
            self.embeddings = None

    def token_embedding_layer(self, tokens, input_mask):
        if self.word_embed_type == 'bert':
            # the original BERT model cannot to use placeholder self.is_training. Set to False: no dropout in bert model!
            bert_model = bert_modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=tokens,
                input_mask=input_mask,
                token_type_ids=tf.zeros(tf.shape(tokens), tf.int32),
                use_one_hot_embeddings=False,
                scope='bert'
            )
            return bert_model.get_sequence_output()

        else:
            return tf.nn.embedding_lookup(self.embeddings, tokens)

    def conv(self, feature_in, filter_shape, strides, name, padding):
        assert name, "should name a scope for the conv"
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(
                shape=filter_shape,
                name="weights",
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1),
            )
            bias = tf.get_variable(
                shape=filter_shape[-1],
                name="bias",
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )

            result = tf.nn.conv1d(
                feature_in, weights, stride=strides, padding=padding, name=name
            )

            result = tf.nn.bias_add(result, bias)

            # batch_norm
            result = tf.layers.batch_normalization(result, training=self.is_training)

            # activation
            result = tf.nn.relu(result)

        return result

    def lstm_attention(self, inputs, actual_length, initial_state=None):
        batch_size = tf.shape(inputs)[0]
        pad_step_embedded = tf.zeros([batch_size, self.lstm_hidden_size * 2], dtype=tf.float32)

        def initial_fn():
            initial_elements_finished = (0 >= actual_length)
            initial_input = inputs[:, 0, :]  # use encoder output only
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # just select the highest logit
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            next_input = inputs[:, time, :]
            elements_finished = (time >= actual_length)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_input = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            next_state = state
            return elements_finished, next_input, next_state

        my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.lstm_hidden_size, memory=inputs,
                memory_sequence_length=actual_length)
            cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_hidden_size * 2)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=self.lstm_hidden_size)

            if initial_state is None:
                initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=my_helper, initial_state=initial_state)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.max_input_len
            )
            return final_outputs, final_state

    def sentence_embedding_model(self, token_embedding, sentence_mask, embedding_type, embedding_size, name=''):
        if name is not None and len(name) > 0:
            name_appendix = '_' + name
        else:
            name_appendix = ''

        if embedding_type == 'CNN':
            with tf.variable_scope("CNN" + name_appendix, reuse=tf.AUTO_REUSE):
                # cascaded CNN
                layer_in = token_embedding
                NumChanelIn = self.word_embed_size
                for layer in range(self.CCNN_NumLayer):
                    conv_out = self.conv(
                        layer_in,
                        [self.CCNN_FilterSize[layer],
                         NumChanelIn,
                         self.CCNN_ChannelSize[layer],
                         ],
                        self.CCNN_ConvStride[layer],
                        "conv_%d" % layer,
                        "SAME",
                    )

                    if layer == self.CCNN_NumLayer - 1 and self.CCNN_PoolSize[layer] == -1:
                        # Max pooling along the whole time axis
                        # Since the kernel size for the following full-connected layer has to be determined in the
                        # initialization stage, the output size of the final conv layer has to be of fixed length, which
                        # make arbitrary pooling size and dynamic sentence length not possible.
                        # Note: max_pooling1d doesn't support dynamic pool_size
                        layer_out = tf.reduce_max(
                            conv_out, axis=1, keep_dims=False,
                            name="max_pool_%d" % layer)
                    else:
                        layer_out = tf.layers.max_pooling1d(
                            conv_out, self.CCNN_PoolSize[layer], self.CCNN_PoolSize[layer], padding='valid',
                            name="max_pool_%d" % layer)

                    # for next layer inputs
                    layer_in = layer_out
                    NumChanelIn = self.CCNN_ChannelSize[layer]

                with tf.variable_scope("full_connect", reuse=tf.AUTO_REUSE):
                    if self.CCNN_PoolSize[-1] == -1:
                        fc_in = layer_out
                    else:
                        # If the PoolSize for the final conv layer is specified, assume the input length is fixed
                        # at max_input_len, as a result it cannot support dynamic padding length.
                        fc_in = tf.reshape(
                            layer_out,
                            [-1, int(self.max_input_len / np.prod(self.CCNN_PoolSize)) * layer_out.get_shape()[2]]
                        )

                    weights = tf.get_variable(
                        name="fc_weights", shape=[fc_in.get_shape()[-1], embedding_size]
                    )

                    bias = tf.get_variable(name="fc_bias", shape=[embedding_size])

                    fc_in = tf.layers.dropout(fc_in, rate=self.dropout_rate, training=self.is_training)
                    fc_out = tf.matmul(fc_in, weights) + bias

                    if self.CCNN_FcActivation.lower() == 'tanh':
                        fc_out = tf.tanh(fc_out)
                    elif self.CCNN_FcActivation.lower() != 'none':
                        raise ValueError("Unsupported activation: %s" % self.CCNN_FcActivation)

        elif embedding_type.endswith('PoolDense'):
            with tf.variable_scope("PoolDense" + name_appendix, reuse=tf.AUTO_REUSE):
                if embedding_type == 'MaxPoolDense':
                    pool_out = tf.reduce_max(token_embedding - 100 * tf.expand_dims(1 - sentence_mask, -1),
                                             axis=1, keepdims=False)
                elif embedding_type == 'AvgPoolDense':
                    pool_out = tf.reduce_mean(token_embedding - 100 * tf.expand_dims(1 - sentence_mask, -1),
                                              axis=1, keepdims=False)
                elif embedding_type == 'AvgMaxPoolDense':
                    pool_out = tf.concat([tf.reduce_mean(token_embedding - 100 * tf.expand_dims(1 - sentence_mask, -1),
                                                         axis=1, keepdims=False),
                                          tf.reduce_max(token_embedding - 100 * tf.expand_dims(1 - sentence_mask, -1),
                                                        axis=1, keepdims=False)],
                                         axis=-1)
                elif embedding_type == 'FirstPoolDense':
                    # use the embedding of the first token only
                    # same as BERT model processing for classification, in which the first token is [cls].
                    pool_out = token_embedding[:, 0]
                else:
                    raise NotImplementedError

                fc_in = tf.layers.dropout(pool_out, rate=self.dropout_rate, training=self.is_training)
                fc_out = tf.layers.dense(
                    fc_in,
                    embedding_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))

        elif embedding_type == 'BiLSTM':
            # Bi-LSTM Layer
            with tf.variable_scope("BiLSTM" + name_appendix, reuse=tf.AUTO_REUSE):
                actual_length = tf.cast(tf.reduce_sum(sentence_mask, axis=1), tf.int32)
                lstm_in = token_embedding
                for layer in range(self.lstm_num_layer):
                    with tf.variable_scope("layer_" + str(layer), reuse=tf.AUTO_REUSE):
                        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size)  # forward direction cell
                        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size)  # backward direction cell

                        keep_prob = tf.cond(self.is_training,
                                            lambda: tf.constant(1 - self.dropout_rate), lambda: tf.constant(1.0))
                        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
                        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)
                        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                            lstm_fw_cell, lstm_bw_cell, lstm_in, sequence_length=actual_length, dtype=tf.float32)
                        # Concat output, [batch_size, sequence_length, lstm_hidden_size * 2]
                        lstm_out = tf.concat((outputs_fw, outputs_bw), axis=2)

                        if self.lstm_attention_enable:
                            # use attention based decoder
                            decoder_output, decoder_state = self.lstm_attention(lstm_out, actual_length)
                            lstm_in = decoder_output.rnn_output
                        else:
                            lstm_in = lstm_out

                if self.lstm_attention_enable:
                    fc_in = tf.concat((decoder_state.cell_state.c, decoder_state.cell_state.h), 1)
                else:
                    final_state_c = tf.concat((final_state_fw.c, final_state_bw.c), 1)
                    final_state_h = tf.concat((final_state_fw.h, final_state_bw.h), 1)
                    fc_in = tf.concat(rnn.LSTMStateTuple(c=final_state_c, h=final_state_h), 1)

                fc_in = tf.layers.dropout(fc_in, rate=self.dropout_rate, training=self.is_training)

                # Fully Connected Layer
                with tf.variable_scope("full_connect", reuse=tf.AUTO_REUSE):
                    fc_out = tf.layers.dense(
                        fc_in,
                        embedding_size,
                        activation=tf.tanh,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))

        # normalize
        embedding = tf.nn.l2_normalize(fc_out, axis=1)

        return embedding

    def build_graph(self):
        batch_size = tf.shape(self.input_sentence)[0]
        input_len = tf.shape(self.input_sentence)[1]
        num_sentence = tf.shape(self.input_sentence_mapping)[2]

        # decrease the logits for padding and query tokens
        contex_sentences_mask = tf.reduce_sum(self.input_sentence_mapping, axis=2)

        # for [CLS] token
        first_token_mask = tf.cast(tf.range(input_len) < 1, tf.float32) + tf.zeros([batch_size, input_len])

        # --------------------------------------------------------------------------------------------------------------
        # token embedding layer

        # get the word embeddings
        token_embedding = self.token_embedding_layer(self.input_sentence, self.input_mask)

        # --------------------------------------------------------------------------------------------------------------
        # sentence embedding layer

        # query sentence length, including [CLS] token, excluding 2 [SEP] tokens
        query_sentence_length = tf.cast(tf.reduce_sum(self.input_mask - contex_sentences_mask, axis=1, keepdims=True),
                                        tf.int32) - 2
        query_mask = tf.cast(tf.expand_dims(tf.range(input_len), axis=0) < query_sentence_length, tf.float32)
        query_mask = query_mask - first_token_mask

        # sentence-token mapping including query and context sentences
        # batch_size x input_length x (1+num_sentence)
        all_sentence_mapping = tf.concat([tf.expand_dims(query_mask, axis=2),
                                          tf.cast(self.input_sentence_mapping, tf.float32)],
                                         axis=2)

        sentence_token_embedding_use_full_dimension = False
        if sentence_token_embedding_use_full_dimension or self.sentence_embedding_type.endswith('PoolDense'):
            # represent each sentence's token embedding at full demenstion (input_length)
            # This consumes too much memory.
            # And for BiLSTM, this doens't work, since it requires the real tokens to start at the begining.
            # For CNN, it's not using sentence mask explictly.

            # batch_size x input_length x (1+num_sentence) x hidden_size
            sentence_token_embedding = tf.matmul(tf.expand_dims(all_sentence_mapping, 3),
                                                 tf.expand_dims(token_embedding, 2))

            if not self.sentence_embedding_type.endswith('PoolDense'):
                # reshape to: (batch_size*(1+num_sentence)) x input_length x hidden_size
                sentence_token_embedding = tf.reshape(tf.transpose(sentence_token_embedding, perm=[0, 2, 1, 3]),
                                                      (-1, input_len, self.word_embed_size))
                sentence_mask = tf.reshape(tf.transpose(all_sentence_mapping, perm=[0, 2, 1]), (-1, input_len))
            else:
                # the reshaping is not need for 'PoolDense' embedding types, since they can process 4-D array (always
                # pooling on the 2nd dimenstion)
                sentence_mask = all_sentence_mapping
        else:
            # reduce each sentence's token embedding sequence length, to the maximum sentence length

            # batch_size x (1+num_sentence)
            sentence_lengths = tf.cast(tf.reduce_sum(all_sentence_mapping, axis=1), tf.int32)
            max_sent_len = tf.reduce_max(sentence_lengths)

            def sample_loop_cond(sample_idx, sample_ary):
                return tf.less(sample_idx, batch_size)

            def sample_loop_body(sample_idx, sample_ary):

                def sentence_loop_cond(sentence_idx, sentence_ary):
                    return tf.less(sentence_idx, num_sentence+1)

                def sentence_loop_body(sentence_idx, sentence_ary):
                    # sentence_length x hidden_size
                    sample_sentence_token_embedding = tf.gather(
                        token_embedding[sample_idx],
                        tf.squeeze(tf.where(all_sentence_mapping[sample_idx, :, sentence_idx] > 0), axis=1))
                    padding = tf.zeros((max_sent_len - sentence_lengths[sample_idx, sentence_idx],
                                        self.word_embed_size))
                    # max_sent_len x hidden_size
                    sample_sentence_token_embedding = tf.concat([sample_sentence_token_embedding, padding], axis=0)

                    return sentence_idx + 1, sentence_ary.write(sentence_idx, sample_sentence_token_embedding)

                sentence_embedding_ary = tf.TensorArray(tf.float32, size=num_sentence+1, infer_shape=True)
                _, sentence_embedding_ary = tf.while_loop(
                    sentence_loop_cond,
                    sentence_loop_body,
                    (tf.constant(0), sentence_embedding_ary),
                    name='sentence_reconstructor'
                )
                sentence_embeddings = sentence_embedding_ary.stack()

                return sample_idx + 1, sample_ary.write(sample_idx, sentence_embeddings)

            embedding_ary = tf.TensorArray(tf.float32, size=batch_size, infer_shape=True)
            _, embedding_ary = tf.while_loop(
                sample_loop_cond,
                sample_loop_body,
                (tf.constant(0), embedding_ary),
                name='batch_reconstructor'
            )
            sentence_token_embedding = embedding_ary.stack()

            # (batch_size*(1+num_sentence)) x max_sent_len x hidden_size
            sentence_token_embedding = tf.reshape(
                sentence_token_embedding, (batch_size * (num_sentence + 1), max_sent_len, self.word_embed_size))

            sentence_mask = tf.cast(
                tf.expand_dims(tf.range(max_sent_len), axis=0) < tf.reshape(sentence_lengths, (-1, 1)),
                tf.float32)

        sentence_embedding = self.sentence_embedding_model(sentence_token_embedding, sentence_mask,
                                                           embedding_type=self.sentence_embedding_type,
                                                           embedding_size=self.sentence_embed_size,
                                                           name='support_fact_sentence')

        if sentence_embedding.get_shape().ndims == 2:
            # reshape to: batch_size x (1+num_sentence) x sentence_embed_size
            sentence_embedding = tf.reshape(sentence_embedding, (batch_size, -1, self.sentence_embed_size))

        # --------------------------------------------------------------------------------------------------------------
        # support fact reasoning layer

        if self.support_fact_reasoning_model == 'Transformer':
            with tf.variable_scope("support_fact_model", reuse=tf.AUTO_REUSE):
                # batch_size x (1+num_sentence)
                all_sentences = tf.reduce_any(all_sentence_mapping > 0, axis=1)
                # batch_size x (1+num_sentence) x (1+num_sentence)
                sentence_attention_mask = tf.logical_and(tf.expand_dims(all_sentences, axis=2),
                                                         tf.expand_dims(all_sentences, axis=1))

                support_fact_embedding = bert_modeling.transformer_model(
                    sentence_embedding,
                    is_training=self.is_training,
                    attention_mask=sentence_attention_mask,
                    hidden_size=self.sentence_embed_size,
                    num_hidden_layers=self.support_fact_transformer['num_hidden_layers'],
                    num_attention_heads=self.support_fact_transformer['num_attention_heads'],
                    intermediate_size=self.support_fact_transformer['intermediate_size'],
                    intermediate_act_fn=bert_modeling.get_activation(self.support_fact_transformer['hidden_act']),
                    hidden_dropout_prob=self.support_fact_transformer['hidden_dropout_prob'],
                    attention_probs_dropout_prob=self.support_fact_transformer['attention_probs_dropout_prob'],
                    initializer_range=0.02,
                    do_return_all_layers=False
                )

        else:
            # 'None'
            support_fact_embedding = sentence_embedding

        # --------------------------------------------------------------------------------------------------------------
        # support fact prediction layer

        # exclude query sentence for support fact logit calculation
        support_fact_logits = tf.layers.dense(
            support_fact_embedding[:, 1:, :],
            1,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
        )

        # batch_size x num_sentence
        support_fact_logits = tf.squeeze(support_fact_logits, axis=2)

        # Mask the real sentences. For non sentence position, the cross-entropy will be 0.
        valid_sentence_mask = tf.cast(tf.reduce_any(self.input_sentence_mapping > 0, axis=1), tf.float32)
        support_fact_logits = support_fact_logits - 100 * (1 - valid_sentence_mask)

        self.support_fact_prob = tf.sigmoid(support_fact_logits, name='support_fact_prob')

        # only consier answer type not 'unknown'? use labled or predicted type?
        # maybe no need, since the train data has valid support fact label (empty) for 'unknown' case
        support_fact_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_support_facts,
                                                                  logits=support_fact_logits)

        if self.support_fact_loss_all_samples:
            if self.support_fact_loss_type == 'Mean':
                self.support_fact_loss = tf.divide(tf.reduce_sum(support_fact_ce), tf.reduce_sum(valid_sentence_mask),
                                                   name='support_fact_loss')
            elif self.support_fact_loss_type == 'Sum':
                self.support_fact_loss = tf.divide(tf.reduce_sum(support_fact_ce), tf.cast(batch_size, tf.float32),
                                                   name='support_fact_loss')
        else:
            num_valid_sentence = tf.reduce_sum(valid_sentence_mask, axis=1)
            support_fact_samples = tf.squeeze(tf.where(num_valid_sentence > 1), axis=1)
            support_fact_ce_valid = tf.gather(support_fact_ce, support_fact_samples)
            support_fact_valid_sample_number = tf.cast(tf.size(support_fact_samples), tf.float32)
            if self.support_fact_loss_type == 'Mean':
                support_fact_loss = tf.cond(tf.equal(support_fact_valid_sample_number, 0),
                                            lambda: tf.constant(0, tf.float32),
                                            lambda: tf.divide(tf.reduce_sum(support_fact_ce_valid),
                                                              support_fact_valid_sample_number))
            elif self.support_fact_loss_type == 'Sum':
                num_support_fact_sentence = tf.gather(num_valid_sentence, support_fact_samples)
                support_fact_loss = tf.cond(tf.equal(support_fact_valid_sample_number, 0),
                                            lambda: tf.constant(0, tf.float32),
                                            lambda: tf.divide(tf.reduce_sum(support_fact_ce_valid),
                                                              tf.reduce_sum(num_support_fact_sentence)))
            self.support_fact_loss = tf.identity(support_fact_loss, name='support_fact_loss')

        # --------------------------------------------------------------------------------------------------------------
        # answer type and span prediction layer
        if self.answer_pred_use_support_fact_embedding:
            # concatenate token embeddings with copied sentence embedding after support fact reasoning
            token_embedding_ext = tf.concat([token_embedding,
                                             tf.matmul(all_sentence_mapping, support_fact_embedding)],
                                            axis=2)
        else:
            token_embedding_ext = token_embedding

        if self.answer_span_predict_model == 'Transformer':
            with tf.variable_scope("answer_span_model", reuse=tf.AUTO_REUSE):
                # batch_size x input_length x input_length
                token_attention_mask = tf.logical_and(tf.expand_dims(self.input_mask > 0, axis=2),
                                                      tf.expand_dims(self.input_mask > 0, axis=1))

                span_embedding = bert_modeling.transformer_model(
                    token_embedding_ext,
                    is_training=self.is_training,
                    attention_mask=token_attention_mask,
                    hidden_size=self.answer_span_transformer['hidden_size'],
                    num_hidden_layers=self.answer_span_transformer['num_hidden_layers'],
                    num_attention_heads=self.answer_span_transformer['num_attention_heads'],
                    intermediate_size=self.answer_span_transformer['intermediate_size'],
                    intermediate_act_fn=bert_modeling.get_activation(self.answer_span_transformer['hidden_act']),
                    hidden_dropout_prob=self.answer_span_transformer['hidden_dropout_prob'],
                    attention_probs_dropout_prob=self.answer_span_transformer['attention_probs_dropout_prob'],
                    initializer_range=0.02,
                    do_return_all_layers=False
                )

        else:
            # 'None'
            span_embedding = token_embedding_ext

        span_logits = tf.layers.dense(
            span_embedding,
            2,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
        )

        if self.span_loss_all_samples:
            # keep the position on first token
            span_contex_sentences_mask = contex_sentences_mask + first_token_mask
        else:
            span_contex_sentences_mask = contex_sentences_mask

        span_logits = span_logits - 100 * (1 - tf.expand_dims(span_contex_sentences_mask, axis=2))

        # batch_size x input_length x input_length
        span_logit_sum = tf.expand_dims(span_logits[:, :, 0], axis=2) + tf.expand_dims(span_logits[:, :, 1], axis=1)
        span_mask = tf.constant(
            np.tril(np.triu(np.ones((self.max_input_len, self.max_input_len)), 0), self.max_answer_len),
            dtype=tf.float32)
        span_logit_sum = span_logit_sum - 100 * tf.expand_dims(1 - span_mask[:input_len, :input_len], 0)
        self.span_start_pos = tf.arg_max(tf.reduce_max(span_logit_sum, axis=2), dimension=1, name='span_start_pos')
        self.span_end_pos = tf.arg_max(tf.reduce_max(span_logit_sum, axis=1), dimension=1, name='span_end_pos')

        # softmax over all tokens of sentence length
        self.span_start_prob = tf.nn.softmax(span_logits[:, :, 0], name='span_start_prob')
        self.span_end_prob = tf.nn.softmax(span_logits[:, :, 1], name='span_end_prob')
        span_start_ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_span_start, input_len),
                                                                logits=span_logits[:, :, 0])
        span_end_ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_span_end, input_len),
                                                              logits=span_logits[:, :, 1])
        span_ce = span_start_ce + span_end_ce

        if self.span_loss_all_samples:
            self.span_loss = tf.reduce_mean(span_ce, name='span_loss')
        else:
            span_answer_samples = tf.squeeze(tf.where(tf.equal(self.input_answer_type, 0)), axis=1)
            span_loss = tf.cond(tf.equal(tf.size(span_answer_samples), 0),
                                lambda: tf.constant(0, tf.float32),
                                lambda: tf.reduce_mean(tf.gather(span_ce, span_answer_samples)))
            self.span_loss = tf.identity(span_loss, name='span_loss')

        if self.answer_type_use_query_embedding_only:
            # use the query sentence embedding output of support fact reasoning layer
            answer_type_embedding = support_fact_embedding[:, 0, :]
        else:
            answer_type_embedding = self.sentence_embedding_model(span_embedding, self.input_mask,
                                                                  embedding_type=self.answer_type_embedding_type,
                                                                  embedding_size=self.answer_type_embed_size,
                                                                  name='answer_type')
        answer_type_logits = tf.layers.dense(
            answer_type_embedding,
            self.num_answer_type,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
        )

        self.answer_type_prob = tf.nn.softmax(answer_type_logits, name='answer_type_prob')

        self.answer_type_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_answer_type, self.num_answer_type),
                                                    logits=answer_type_logits),
            name='answer_type_loss')

        # --------------------------------------------------------------------------------------------------------------
        # define loss

        loss = (self.span_loss_weight * self.span_loss +
                self.answer_type_loss_weight * self.answer_type_loss +
                self.support_fact_loss_weight * self.support_fact_loss)

        reg_vars = [v for v in self.get_trainable_variables() if v.name.find('bias') == -1]
        reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_vars)
        if self.L2_Normalize:
            self.reg_loss = tf.divide(reg_loss,
                                      float(int(sum([np.prod(v.shape) for v in reg_vars]))),
                                      name='reg_loss')
        else:
            self.reg_loss = tf.identity(reg_loss, name='reg_loss')

        if self.optimizer == 'BertAdam':
            self.loss = tf.identity(loss, name="loss")

        else:
            self.loss = tf.add_n([loss, self.reg_loss], name="loss")

        if self.word_embed_type == 'bert' and self.bert.get('init_checkpoint'):
            # load the pretrained bert model parameters
            (assignment_map, initialized_variable_names
             ) = bert_modeling.get_assignment_map_from_checkpoint(
                tf.trainable_variables(), self.bert['init_checkpoint'])
            tf.train.init_from_checkpoint(self.bert['init_checkpoint'], assignment_map)

        # bert_params = sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('bert')])
        # lstm_params = sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('BiLSTM')])
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        num_trainable_params = sum([np.prod(v.shape) for v in self.get_trainable_variables()])
        print('total number of parameters: %d (%d trainable)' % (num_params, num_trainable_params))

    def get_trainable_variables(self):
        tvars = tf.trainable_variables()
        if self.word_embed_type == 'bert':
            if self.word_embed_trainable:
                # filter out bert frozen layers
                tvars = [v for v in tvars
                         if not np.any([v.name.startswith(frozen_layer) for frozen_layer in self.bert_frozen_layers])]
            else:
                # filter out bert model variables if it's not trainable
                tvars = [v for v in tvars if not v.name.startswith('bert')]
        return tvars

    def define_loss_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.train.exponential_decay(
            self.lr_base,
            self.global_step,
            self.lr_decay_steps,
            self.lr_decay_rate,
            staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.lr_min, name='learning_rate')

        # define the optimizer
        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=self.momentum_rate, beta2=self.momentum_rate_2nd)

        elif self.optimizer == 'BertAdam':
            reg_vars = [v for v in self.get_trainable_variables() if v.name.find('bias') == -1]
            if self.L2_Normalize:
                weight_decay_rate = tf.divide(self.L2_REG_LAMBDA,
                                              float(int(sum([np.prod(v.shape) for v in reg_vars]))))
            else:
                weight_decay_rate = self.L2_REG_LAMBDA
            optimizer = optimization.AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=weight_decay_rate,
                beta_1=self.momentum_rate,
                beta_2=self.momentum_rate_2nd,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        grad_and_vars = optimizer.compute_gradients(self.loss, var_list=self.get_trainable_variables())

        # for grad, var in grad_and_vars:
        #     if var is None or grad is None:
        #         print(var, grad)
        # bert pooler layer is not used, its dense weith/bias has no gradient

        # print(grad_and_vars)
        clipped_glvs = [(tf.clip_by_value(grad, self.grad_threshold * (-1.0), self.grad_threshold), var)
                        for grad, var in grad_and_vars if grad is not None]

        # global_step will be increased for each time running train_op
        train_op = optimizer.apply_gradients(clipped_glvs, global_step=self.global_step)

        if self.optimizer == 'BertAdam':
            # BertAdam is not updating global_step
            new_global_step = self.global_step + 1
            train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])

        # group batch normalization with train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([train_op, update_ops], name='train_op')

    def episode_iter(self, data_set, episode_size=None, shuffle=True, max_input_len=None, keep_invalid=True):
        """Generate batch vec_data. """
        if shuffle:
            random.shuffle(data_set)
        num_sample = len(data_set)
        if episode_size is None:
            episode_size = self.batch_size
        elif episode_size == -1:
            episode_size = num_sample
        num_episode = int((num_sample - 1) / episode_size) + 1

        # 0 corresponds to [CLS] token for BERT tokenizer
        IGNORE_INDEX = 0
        for i in range(num_episode):
            start_id = i * episode_size
            end_id = min((i + 1) * episode_size, num_sample)
            cur_bsz = end_id - start_id
            cur_batch = data_set[start_id:end_id]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            max_sent_cnt = 0

            context_idxs = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            context_mask = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            segment_idxs = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)

            query_mapping = np.zeros((cur_bsz, self.max_input_len), dtype=np.int32)
            start_mapping = np.zeros((cur_bsz, self.max_num_sentence, self.max_input_len), dtype=np.int32)
            all_mapping = np.zeros((cur_bsz, self.max_input_len, self.max_num_sentence), dtype=np.int32)

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

                for j in range(case.sent_spans[0][0] - 1):
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
                    y1[sample_idx] = IGNORE_INDEX
                    y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 1
                elif case.ans_type == 2:
                    y1[sample_idx] = IGNORE_INDEX
                    y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 2
                elif case.ans_type == 3:
                    y1[sample_idx] = IGNORE_INDEX
                    y2[sample_idx] = IGNORE_INDEX
                    q_type[sample_idx] = 3

                for j, sent_span in enumerate(case.sent_spans[:self.max_num_sentence]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start < end:
                        if q_type[sample_idx] != 3:
                            # set support fact flag only for non-unknown types
                            is_support[sample_idx, j] = int(is_sp_flag)
                        all_mapping[sample_idx, start:end + 1, j] = 1
                        start_mapping[sample_idx, j, start] = 1

                if q_type[sample_idx] != 3 and np.sum(is_support[sample_idx]) == 0:
                    # set to invalid if support fact is missing
                    is_valid = False

                if is_valid or keep_invalid:
                    ids.append(case.qas_id)
                    max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))
                    valid_sample_idx.append(sample_idx)

            if len(valid_sample_idx) == 0:
                continue

            valid_idx = np.array(valid_sample_idx)
            input_lengths = np.sum(context_mask[valid_idx] > 0, axis=1)
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
                'start_mapping': start_mapping[valid_idx, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[valid_idx, :max_c_len, :max_sent_cnt],

                'is_support': is_support[valid_idx, :max_sent_cnt],
            }

    def fit(self, train_set, validation_set, epochs):

        if self.LoadedModel:
            out_dir = os.path.dirname(os.path.dirname(self.LoadedModel))
        else:
            self.initialize_session()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(os.path.curdir, 'runs', 'rc_' + timestamp)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            copyfile(os.path.join(os.path.curdir, 'configs/reading_comprehension_config.yml'),
                     os.path.join(out_dir, 'reading_comprehension_config.yml'))

        with self.session.graph.as_default():
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            best_checkpoint_dir = os.path.join(out_dir, "bestcheckpoints")
            saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=True)

            # Train summaries
            summary_dir = os.path.join(out_dir, "summaries")
            summary_writer = tf.summary.FileWriter(summary_dir, self.session.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)

            for epoch in range(1, epochs + 1):
                for batch in self.episode_iter(train_set, keep_invalid=False):
                    (span_start_pos, span_end_pos, span_start_prob, span_end_prob, answer_type_prob, support_fact_prob,
                     span_loss, answer_type_loss, support_fact_loss, reg_loss, loss, lr, _, global_step
                     ) = self.session.run(
                        [self.span_start_pos, self.span_end_pos, self.span_start_prob, self.span_end_prob,
                         self.answer_type_prob, self.support_fact_prob,
                         self.span_loss, self.answer_type_loss, self.support_fact_loss, self.reg_loss, self.loss,
                         self.learning_rate, self.train_op, self.global_step],
                        feed_dict={
                            self.input_sentence: batch['context_idxs'],
                            self.input_mask: batch['context_mask'],
                            self.input_answer_type: batch['q_type'],
                            self.input_span_start: batch['y1'],
                            self.input_span_end: batch['y2'],
                            self.input_sentence_mapping: batch['all_mapping'],
                            self.input_support_facts: batch['is_support'],
                            self.is_training: True
                        }
                    )

                    if np.isnan(loss):
                        print(batch)
                        raise SystemError

                    if self.restrict_answer_span:
                        span_start_predicts, span_end_predicts = self.predict_answer_span(
                            batch['context_idxs'], span_start_prob, span_end_prob)
                    else:
                        span_start_predicts = span_start_pos
                        span_end_predicts = span_end_pos

                    answer_type_predicts, support_fact_predicts = self.predict_answer_type_and_support_fact(
                        answer_type_prob, support_fact_prob)

                    (answer_type_accu,
                     span_iou,
                     answer_score,
                     support_fact_accu,
                     support_fact_recall,
                     support_fact_precision,
                     support_fact_f1,
                     joint_metric,
                     num_span,
                     num_support_fact
                     ) = self.evaluate(
                        {
                            'span_start_pos': span_start_predicts,
                            'span_end_pos': span_end_predicts,
                            'answer_type': answer_type_predicts,
                            'support_fact': support_fact_predicts
                        },
                        batch
                    )

                    print("Epoch: {}\tCount: {}\tspan_loss:{:.4f}\tanswer_type_loss:{:.4f}\t"
                          "support_fact_loss:{:.4f}\treg_loss:{:.4f}\tloss:{:.4f}\tanswer_score:{:.4f}\t"
                          "support_fact_f1:{:.4f}\tjoint_metric:{:.4f}".format(
                            epoch, global_step, span_loss, answer_type_loss, support_fact_loss, reg_loss, loss,
                            answer_score, support_fact_f1, joint_metric))

                    train_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="span_loss",
                                         simple_value=span_loss),
                        tf.Summary.Value(tag="answer_type_loss",
                                         simple_value=answer_type_loss),
                        tf.Summary.Value(tag="support_fact_loss",
                                         simple_value=support_fact_loss),
                        tf.Summary.Value(tag="reg_loss",
                                         simple_value=reg_loss),
                        tf.Summary.Value(tag="train_loss",
                                         simple_value=loss),
                        tf.Summary.Value(tag="learning_rate",
                                         simple_value=lr),
                        tf.Summary.Value(tag="answer_type_accu",
                                         simple_value=answer_type_accu),
                        tf.Summary.Value(tag="span_iou",
                                         simple_value=span_iou),
                        tf.Summary.Value(tag="answer_score",
                                         simple_value=answer_score),
                        tf.Summary.Value(tag="support_fact_accu",
                                         simple_value=support_fact_accu),
                        tf.Summary.Value(tag="support_fact_recall",
                                         simple_value=support_fact_recall),
                        tf.Summary.Value(tag="support_fact_precision",
                                         simple_value=support_fact_precision),
                        tf.Summary.Value(tag="support_fact_f1",
                                         simple_value=support_fact_f1),
                        tf.Summary.Value(tag="joint_metric",
                                         simple_value=joint_metric)
                    ])
                    summary_writer.add_summary(train_summary, global_step)

                    if global_step % self.save_period == 0:
                        saver.save(self.session, checkpoint_dir + "/model", global_step=self.global_step)

                    if global_step % self.validate_period == 0:
                        num_val_sets = int((len(validation_set) - 1) / self.validate_size) + 1
                        val_set_idx = int(global_step / self.validate_period) % num_val_sets
                        val_set_start = val_set_idx * self.validate_size
                        val_set_end = min((val_set_idx + 1) * self.validate_size, len(validation_set))

                        (val_span_loss, val_answer_type_loss, val_support_fact_loss, val_loss, val_answer_type_accu,
                         val_span_iou, val_answer_score, val_support_fact_accu, val_support_fact_recall,
                         val_support_fact_precision, val_support_fact_f1, val_joint_metric
                         ) = self.test(validation_set[val_set_start:val_set_end])

                        best_saver.handle(val_joint_metric, self.session, global_step)

                        valid_summary = tf.Summary(value=[
                            tf.Summary.Value(tag="validation_span_loss",
                                             simple_value=val_span_loss),
                            tf.Summary.Value(tag="validation_answer_type_loss",
                                             simple_value=val_answer_type_loss),
                            tf.Summary.Value(tag="validation_support_fact_loss",
                                             simple_value=val_support_fact_loss),
                            tf.Summary.Value(tag="validation_loss",
                                             simple_value=val_loss),
                            tf.Summary.Value(tag="validation_answer_type_accu",
                                             simple_value=val_answer_type_accu),
                            tf.Summary.Value(tag="validation_span_iou",
                                             simple_value=val_span_iou),
                            tf.Summary.Value(tag="validation_answer_score",
                                             simple_value=val_answer_score),
                            tf.Summary.Value(tag="validation_support_fact_accu",
                                             simple_value=val_support_fact_accu),
                            tf.Summary.Value(tag="validation_support_fact_recall",
                                             simple_value=val_support_fact_recall),
                            tf.Summary.Value(tag="validation_support_fact_precision",
                                             simple_value=val_support_fact_precision),
                            tf.Summary.Value(tag="validation_support_fact_f1",
                                             simple_value=val_support_fact_f1),
                            tf.Summary.Value(tag="validation_joint_metric",
                                             simple_value=val_joint_metric)
                        ])
                        summary_writer.add_summary(valid_summary, global_step)

                    summary_writer.flush()

            coord.request_stop()
            coord.join(threads)

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

            answer_dict.update(answer_dict_batch)

            for sample_idx, support_facts in enumerate(support_fact_predicts):
                cur_id = ids[sample_idx]
                support_fact_sentences = [
                    example_dict[cur_id].sent_names[sentence_idx] for sentence_idx in support_facts]
                support_fact_dict.update({cur_id: support_fact_sentences})

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

    def tokenize(self, samples, max_input_len=None):
        if max_input_len is None:
            max_input_len = self.max_input_len

        if self.word_embed_type == 'pretrained':
            if max_input_len == -1:
                max_input_len = max([len(sample) for sample in samples])

            tokenindex = [tokenize(sample, self.vocab) for sample in samples]
            lengths = [len(item) for item in tokenindex]
            tokens = pad_sequences(tokenindex, maxlen=max_input_len, value=0)

        elif self.word_embed_type == 'bert':
            if max_input_len == -1:
                # BERT tokenizer will insert two additional tokens
                max_input_len = max([len(sample) + 2 for sample in samples])
            max_input_len = min(max_input_len, self.bert_config.max_position_embeddings)
            tokens = []
            lengths = np.zeros(len(samples), np.int32)
            for (i, sample) in enumerate(samples):
                guid = '%d' % i
                text_a = tokenization.convert_to_unicode(sample)
                example = run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label='null')
                feature = run_classifier.convert_single_example(
                    0, example, ['null'], max_input_len, self.bert_tokenizer)
                tokens.append(feature.input_ids)
                lengths[i] = sum(feature.input_mask)
            tokens = np.array(tokens)

        return tokens, lengths
