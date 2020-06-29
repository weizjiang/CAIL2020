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

from reading_comprehension.utils import checkmate as cm
from reading_comprehension.utils.pipelines import truncate_by_separator, token_to_index as tokenize

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
        self.max_sentence_len = config.get('max_sentence_len', 16)  # the default max sentence length for padding
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
                vocab_file = os.path.join(os.path.dirname(__file__), 'data', 'vocab.txt')

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

        self.sentence_embedding_type = config.get('sentence_embedding_type', 'CNN')
        self.sentence_embed_size = config.get('sentence_embed_size', 1024)
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


    def initialize_session(self):
        print("start to initialize the variables ")
        print("create session. ")

        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()

            self.regularizer = tf.contrib.layers.l2_regularizer(self.L2_REG_LAMBDA)

            # shape: batch_size x max_sentence_length
            self.input_sentence = tf.placeholder(tf.int32, [None, None], name="input_sentence")
            # shape: batch_size
            self.input_actual_length = tf.placeholder(tf.int32, [None], name='input_actual_length')
            # shape: batch_size x num_answer_type
            self.input_answer_type = tf.placeholder(tf.int32, shape=[None, self.num_answer_type],
                                                    name="input_answer_type")
            # shape: batch_size
            self.input_span_start = tf.placeholder(tf.int32, shape=[None], name="input_span_start")
            # shape: batch_size
            self.input_span_end = tf.placeholder(tf.int32, shape=[None], name="input_span_end")
            # shape: batch_size x num_sentence
            self.input_support_facts = tf.placeholder(tf.int32, shape=[None, None], name="input_support_facts")
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
            initial_input = inputs[:, 0, :] # use encoder output only
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
            cell = tf.contrib.rnn.LSTMCell(num_units=self.lstm_hidden_size*2)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=self.lstm_hidden_size)

            if initial_state is None:
                initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=my_helper, initial_state=initial_state)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.max_sentence_len
            )
            return final_outputs, final_state

    def sentence_embedding_model(self, token_embedding, actual_length, name=''):
        if name is not None and len(name) > 0:
            name_appendix = '_' + name
        else:
            name_appendix = ''

        if self.sentence_embedding_type == 'CNN':
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
                        # at max_sentence_len, as a result it cannot support dynamic padding length.
                        fc_in = tf.reshape(
                            layer_out,
                            [-1, int(self.max_sentence_len/np.prod(self.CCNN_PoolSize)) * layer_out.get_shape()[2]]
                        )

                    weights = tf.get_variable(
                        name="fc_weights", shape=[fc_in.get_shape()[-1], self.sentence_embed_size]
                    )

                    bias = tf.get_variable(name="fc_bias", shape=[self.sentence_embed_size])

                    fc_in = tf.layers.dropout(fc_in, rate=self.dropout_rate, training=self.is_training)
                    fc_out = tf.matmul(fc_in, weights) + bias

                    if self.CCNN_FcActivation.lower() == 'tanh':
                        fc_out = tf.tanh(fc_out)
                    elif self.CCNN_FcActivation.lower() != 'none':
                        raise ValueError("Unsupported activation: %s" % self.CCNN_FcActivation)

        elif self.sentence_embedding_type == 'PoolDense':
            with tf.variable_scope("PoolDense" + name_appendix, reuse=tf.AUTO_REUSE):
                # use the embedding of the first token only, and pass it to a dense layer
                # same as BERT model processing for classification, in which the first token is [cls].
                first_token_tensor = tf.squeeze(token_embedding[:, 0:1, :], axis=1)
                fc_in = tf.layers.dropout(first_token_tensor, rate=self.dropout_rate, training=self.is_training)
                fc_out = tf.layers.dense(
                    fc_in,
                    self.sentence_embed_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))

        elif self.sentence_embedding_type == 'BiLSTM':
            # Bi-LSTM Layer
            with tf.variable_scope("BiLSTM" + name_appendix, reuse=tf.AUTO_REUSE):
                lstm_in = token_embedding
                for layer in range(self.lstm_num_layer):
                    with tf.variable_scope("layer_" + str(layer), reuse=tf.AUTO_REUSE):
                        lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size)  # forward direction cell
                        lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_hidden_size)  # backward direction cell

                        keep_prob = tf.cond(self.is_training,
                                            lambda: tf.constant(1-self.dropout_rate), lambda: tf.constant(1.0))
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
                        self.sentence_embed_size,
                        activation=tf.tanh,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))

        # normalize
        embedding = tf.nn.l2_normalize(fc_out, axis=1)

        return embedding

    def define_score_loss(self, sentence_embedding):
        if self.sentence_embedding_shared_by_scores:
            dense_out = tf.layers.dense(
                sentence_embedding,
                self.num_score,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
            )

        else:
            dense_out = None
            for sentence_embed in sentence_embedding:
                dense_out_per_score = tf.layers.dense(
                    sentence_embed,
                    1,
                    activation=None,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
                )
                if dense_out is None:
                    dense_out = dense_out_per_score
                else:
                    dense_out = tf.concat([dense_out, dense_out_per_score], axis=1)

        self.output_scores = tf.sigmoid(dense_out, name='output_scores')

        # calculate MSE

        # # samples in the batch may use different score type, averaged MSE
        # self.score_mse = tf.reduce_mean(tf.square(tf.subtract(
        #     tf.gather_nd(self.output_scores, tf.where(tf.greater_equal(self.input_scores, 0.))),
        #     tf.gather_nd(self.input_scores, tf.where(tf.greater_equal(self.input_scores, 0.)))
        # )), name='score_mse')
        # score_loss = self.score_mse

        # # assume the samples in the batch use same score type (alternate training), per-score MSE
        # # self.score_mse = tf.reduce_mean(tf.square(tf.subtract(self.output_scores, self.input_scores)),
        # #                                 axis=0, name='score_mse')
        # score_valid_flag = tf.reduce_any(tf.greater_equal(self.input_scores, 0.), axis=0)
        # score_valid_idx = tf.linalg.tensor_diag(tf.cast(score_valid_flag, tf.float32))
        # self.score_mse = tf.reduce_mean(tf.square(tf.subtract(
        #     tf.matmul(self.output_scores, score_valid_idx),
        #     tf.matmul(self.input_scores, score_valid_idx)
        # )), axis=0, name='score_mse')
        # score_loss = tf.reduce_mean(tf.gather(self.score_mse, tf.where(score_valid_flag)))

        # allow samples in the batch use different score type, per-score MSE
        score_valid_flag_mat = tf.greater_equal(self.input_scores, 0.)
        score_valid_flag = tf.reduce_any(score_valid_flag_mat, axis=0)
        score_valid_mask = tf.cast(score_valid_flag_mat, tf.float32)
        self.score_mse = tf.div_no_nan(
            tf.reduce_sum(
                tf.square(tf.multiply(tf.subtract(self.output_scores, self.input_scores), score_valid_mask)),
                axis=0),
            tf.reduce_sum(score_valid_mask, axis=0),
            name='score_mse')

        if self.score_loss_type == 'MSE':
            score_loss = tf.reduce_mean(tf.gather(self.score_mse, tf.where(score_valid_flag)))

        elif self.score_loss_type == 'RMSE':
            score_loss = tf.sqrt(tf.reduce_mean(tf.gather(self.score_mse, tf.where(score_valid_flag))))

        elif self.score_loss_type == 'CrossEntropy':
            # # calculate cross-entropy, averaging all valid scores
            # score_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     labels=tf.gather_nd(self.input_scores, tf.where(tf.greater_equal(self.input_scores, 0.))),
            #     logits=tf.gather_nd(dense_out, tf.where(tf.greater_equal(self.input_scores, 0.)))
            # ))

            # calculate per-score cross entropy first, for alternate training
            score_ce = tf.div_no_nan(
                tf.reduce_sum(
                    tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_scores,
                                                                        logits=dense_out),
                                score_valid_mask),
                    axis=0),
                tf.reduce_sum(score_valid_mask, axis=0),
                name='score_ce')

            score_loss = tf.reduce_mean(tf.gather(score_ce, tf.where(score_valid_flag)))

        else:
            raise NotImplementedError

        # apply the weight to the distance loss
        self.score_loss = tf.multiply(self.score_loss_weight, score_loss, name='score_loss')

    def build_graph(self):
        batch_size = tf.shape(self.input_sentence)[0]
        sentence_len = tf.shape(self.input_sentence)[1]
        input_mask = tf.cast(tf.less(tf.range(0, sentence_len), tf.reshape(self.input_actual_length, (-1, 1))),
                             tf.float32)

        # get the word embeddings
        token_embedding = self.token_embedding_layer(self.input_sentence, input_mask)

        span_logits = tf.layers.dense(
            token_embedding,
            2,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
        )

        # decreas the logits for padding. Only use context mask?
        span_logits = span_logits - 1e30 * tf.reshape(1 - input_mask, (batch_size, sentence_len, 1))

        # softmax over all tokens of sentence length
        span_start_ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_span_start, sentence_len),
                                                                logits=span_logits[:, :, 0])
        span_end_ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_span_end, sentence_len),
                                                              logits=span_logits[:, :, 1])
        span_ce = span_start_ce + span_end_ce
        span_answer_flag = tf.equal(self.input_answer_type[:, 0], 1)
        span_loss = tf.reduce_mean(tf.gather(span_ce, tf.where(span_answer_flag)), name='span_loss')

        # how to reduce on 1 token?
        type_logits = tf.layers.dense(
            token_embedding,
            self.num_answer_type,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02)
        )

        # type_ce =

        reg_vars = [v for v in self.get_trainable_variables() if v.name.find('bias') == -1]
        reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_vars)
        if self.L2_Normalize:
            self.reg_loss = tf.divide(reg_loss,
                                      float(int(sum([np.prod(v.shape) for v in reg_vars]))),
                                      name='reg_loss')
        else:
            self.reg_loss = tf.identity(reg_loss, name='reg_loss')

        if self.optimizer == 'BertAdam':
            self.loss = tf.identity(self.score_loss, name="loss")

        else:
            self.loss = tf.add_n([self.score_loss, self.reg_loss], name="loss")

        if self.word_embed_type == 'bert' and self.bert.get('init_checkpoint'):
            # load the pretrained bert model parameters
            (assignment_map, initialized_variable_names
             ) = bert_modeling.get_assignment_map_from_checkpoint(
                tf.trainable_variables(), self.bert['init_checkpoint'])
            tf.train.init_from_checkpoint(self.bert['init_checkpoint'], assignment_map)

        # bert_params = sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('bert')])
        # lstm_params = sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('BiLSTM')])
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print('total number of parameters: %d' % num_params)

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


    def episode_iter(self, data_set, episode_size=None, shuffle=True, same_score_type=True, max_sentence_len=None):
        """Generate batch vec_data. """
        if shuffle:
            random.shuffle(data_set)
        num_sentence = len(data_set)
        if episode_size is None:
            episode_size = self.batch_size
        elif episode_size == -1:
            episode_size = num_sentence
        num_episode = int((num_sentence - 1) / episode_size) + 1

        for i in range(num_episode):
            start_id = i * episode_size
            end_id = min((i + 1) * episode_size, num_sentence)
            # cyclically selecting the score type for alternate training
            score_type_idx = i % len(self.score_types) if same_score_type else None
            deteriorated_samples, scores = self.deteriorate_samples(
                data_set[start_id:end_id], score_type_idx, max_sentence_len)
            sample_tokens, sample_token_lengths = self.tokenize(deteriorated_samples, max_sentence_len)

            yield sample_tokens, scores, sample_token_lengths, deteriorated_samples

    def fit(self, train_set, validation_set, epochs):

        if self.LoadedModel:
            out_dir = os.path.dirname(os.path.dirname(self.LoadedModel))
        else:
            self.initialize_session()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(os.path.curdir, 'runs', 'qe_' + timestamp)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            copyfile(os.path.join(os.path.curdir, 'configs/config.yml'), os.path.join(out_dir, 'config.yml'))

        # rloss = 0.0

        with self.session.graph.as_default():
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            best_checkpoint_dir = os.path.join(out_dir, "bestcheckpoints")
            saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=1, maximize=False)

            # Train summaries
            summary_dir = os.path.join(out_dir, "summaries")
            summary_writer = tf.summary.FileWriter(summary_dir, self.session.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        
            for epoch in range(1, epochs + 1):
                episode_data = self.episode_iter(train_set, same_score_type=True)

                for x, y, actual_len, _ in episode_data:
                    (score_loss, reg_loss, loss, lr, _, global_step,
                     ) = self.session.run(
                        [self.score_loss, self.reg_loss, self.loss, self.learning_rate, self.train_op, self.global_step],
                        feed_dict={
                            self.input_sentence: x,
                            self.input_actual_length: actual_len,
                            self.input_scores: y,
                            self.is_training: True
                        }
                    )

                    print("Epoch: {}\t Count: {}\t score_loss:{:.4f}\t reg_loss:{:.4f}\t loss:{:.4f}".format(
                        epoch, global_step, score_loss, reg_loss, loss))

                    train_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="train_score_loss",
                                         simple_value=score_loss),
                        tf.Summary.Value(tag="reg_loss",
                                         simple_value=reg_loss),
                        tf.Summary.Value(tag="train_loss",
                                         simple_value=loss),
                        tf.Summary.Value(tag="learning_rate",
                                         simple_value=lr),
                    ])
                    summary_writer.add_summary(train_summary, global_step)

                    # if rloss == 0.0 or loss < rloss: # * 0.8
                    #     rloss = loss

                    if global_step % self.save_period == 0:
                        saver.save(self.session, checkpoint_dir + "/model", global_step=self.global_step)

                    if global_step % self.validate_period == 0:
                        # only use one batch for validation
                        val_batch_size = self.batch_size
                        num_val_batch = int((len(validation_set) - 1) / val_batch_size) + 1
                        val_batch_idx = int(global_step/self.validate_period) % num_val_batch
                        val_batch_start = val_batch_idx * val_batch_size
                        val_batch_end = min((val_batch_idx+1) * val_batch_size, len(validation_set))
                        validation_data = self.episode_iter(validation_set[val_batch_start:val_batch_end],
                                                            val_batch_size,
                                                            shuffle=False,
                                                            same_score_type=False)
                        for (val_x, val_y, val_acutal_len, _) in validation_data:
                            val_score_loss, val_loss = self.session.run(
                                [self.score_loss, self.loss],
                                feed_dict={
                                    self.input_sentence: val_x,
                                    self.input_actual_length: val_acutal_len,
                                    self.input_scores: val_y,
                                    self.is_training: False
                                }
                            )

                            print(" validation ".center(25, "="))
                            print("score loss: %.4f, loss: %.4f" %
                                  (val_score_loss, val_loss))

                            best_saver.handle(val_score_loss, self.session, global_step)

                            valid_summary = tf.Summary(value=[
                                tf.Summary.Value(tag="validation_score_loss",
                                                 simple_value=val_score_loss),
                                tf.Summary.Value(tag="validation_loss",
                                                 simple_value=val_loss)
                            ])
                            summary_writer.add_summary(valid_summary, global_step)

                    summary_writer.flush()
                    
            coord.request_stop()
            coord.join(threads)

    def load_model(self, model_path, selection='L'):

        if selection == 'B':
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(model_path, 'bestcheckpoints'))
        else:
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(model_path, 'checkpoints'))

        # overwrite the original configuration
        configFile = os.path.join(model_path, 'config.yml')
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
            self.input_actual_length = graph.get_tensor_by_name("input_actual_length:0")
            self.input_scores = graph.get_tensor_by_name("input_scores:0")
            self.is_training = graph.get_tensor_by_name("is_training:0")

            self.output_scores = graph.get_tensor_by_name("output_scores:0")
            self.score_mse = graph.get_tensor_by_name("score_mse:0")
            self.score_loss = graph.get_tensor_by_name("score_loss:0")
            self.reg_loss = graph.get_tensor_by_name("reg_loss:0")
            self.loss = graph.get_tensor_by_name("loss:0")
            self.train_op = graph.get_operation_by_name("train_op")
            self.learning_rate = graph.get_tensor_by_name("learning_rate:0")
            global_steps = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
            if len(global_steps) > 0:
                self.global_step = global_steps[0]

        self.LoadedModel = checkpoint_file

    def test(self, test_set, test_batch_size=None, test_per_sample=False, max_sentence_len=None):
        test_data = self.episode_iter(test_set, test_batch_size, shuffle=False, same_score_type=False,
                                      max_sentence_len=max_sentence_len)
        score_mse_all = np.zeros(len(self.score_types))
        score_loss_all = 0.
        loss_all = 0.
        cnt = 0
        for (test_x, test_y, test_actual_len, test_samples) in test_data:
            val_output_scores, val_score_mse, val_score_loss, val_loss = self.session.run(
                [self.output_scores, self.score_mse, self.score_loss, self.loss],
                feed_dict={
                    self.input_sentence: test_x,
                    self.input_actual_length: test_actual_len,
                    self.input_scores: test_y,
                    self.is_training: False
                }
            )

            if test_per_sample:
                for idx in range(len(test_samples)):
                    print('-----------')
                    print('raw: {}'.format(test_set[cnt + idx]))
                    print('input: {}'.format(test_samples[idx]))
                    # print("score types: {}".format(self.score_types))
                    print('input scores: {}'.format(test_y[idx]))
                    print('output scores: {}'.format(val_output_scores[idx]))

            score_mse_all += val_score_mse
            score_loss_all += val_score_loss
            loss_all += val_loss
            cnt += 1

        print("==============")
        print("score types: {}".format(self.score_types))
        print("score mse: {}, \nscore loss: {:.4f}, \ntotal loss: {:.4f}".format(
              score_mse_all/cnt, score_loss_all/cnt, loss_all/cnt))

    def predict(self, query_set, batch_size=None, max_sentence_len=None, test_per_sample=True):
        num_sentence = len(query_set)
        if batch_size is None:
            batch_size = self.batch_size
        elif batch_size == -1:
            batch_size = num_sentence
        num_batch = int((num_sentence - 1) / batch_size) + 1

        output_scores = None
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, num_sentence)
            sample_tokens, sample_token_lengths = self.tokenize(query_set[start_id:end_id],
                                                                max_sentence_len=max_sentence_len)

            val_output_scores = self.session.run(
                self.output_scores,
                feed_dict={
                    self.input_sentence: sample_tokens,
                    self.input_actual_length: sample_token_lengths,
                    self.is_training: False
                }
            )

            if test_per_sample:
                for idx in range(start_id, end_id):
                    print('-----------')
                    print('input: {}'.format(query_set[idx]))
                    print("score types: {}".format(self.score_types))
                    print('output scores: {}'.format(val_output_scores[idx]))

            if output_scores is None:
                output_scores = val_output_scores
            else:
                output_scores = np.vstack([output_scores, val_output_scores])

        return output_scores

    def tokenize(self, samples, max_sentence_len=None):
        if max_sentence_len is None:
            max_sentence_len = self.max_sentence_len

        if self.word_embed_type == 'pretrained':
            if max_sentence_len == -1:
                max_sentence_len = max([len(sample) for sample in samples])

            tokenindex = [tokenize(sample, self.vocab) for sample in samples]
            lengths = [len(item) for item in tokenindex]
            tokens = pad_sequences(tokenindex, maxlen=max_sentence_len, value=0)

        elif self.word_embed_type == 'bert':
            if max_sentence_len == -1:
                # BERT tokenizer will insert two additional tokens
                max_sentence_len = max([len(sample) + 2 for sample in samples])
            max_sentence_len = min(max_sentence_len, self.bert_config.max_position_embeddings)
            tokens = []
            lengths = np.zeros(len(samples), np.int32)
            for (i, sample) in enumerate(samples):
                guid = '%d' % i
                text_a = tokenization.convert_to_unicode(sample)
                example = run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label='null')
                feature = run_classifier.convert_single_example(
                    0, example, ['null'], max_sentence_len, self.bert_tokenizer)
                tokens.append(feature.input_ids)
                lengths[i] = sum(feature.input_mask)
            tokens = np.array(tokens)

        return tokens, lengths
