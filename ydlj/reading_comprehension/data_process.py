from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from jieba.posseg import dt
import re
import numpy as np

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type
        self.token_to_orig_map=token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def get_entity_spans(text):
    # get entity spans
    # Note: the end position is included

    # get entities by regex rules
    entity_rules = [
        r'[\u4e00-\u9fa5][某×xXⅩ╳\*]+\d{,2}(?!\d)',  # hidden people name
        r'(?<!\d)((?:(?:19|20)\d{2}年)|(?:0?[1-9]|1[0-2])月|(?:(?:[0-2]?[1-9]|10|20|30|31)日)){1,3}',  # date
        r'(?<!\d)(?:([0-1]?\d|2[0-3])[时点][0-5]?\d分([0-5]?\d秒)?|([0-1]?\d|2[0-3])([:：][0-5]\d){1,2}(?!\d))',  # time
        r'(\d+|(?<!\d)\d{1,3}([, \u3000]{1,2}\d{3})*)(\.\d{,2})?[十百千万亿]*元',  # amount of money
        r'《[\u4e00-\u9fa5\w]{1,50}》',  # document title
        r'字?(?:[（【\(\[]?(?:19|20)\d{2}年?[）】\)\]]?)?第?[\d-]{1,8}号',  # document ID
        r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][×a-zA-Z\d\*]{1,7}',  # hidden car license
    ]
    entity_rule_spans = []

    def is_overlap_with_rule_span(start_pos, end_pos):
        overlap_with_rule_span = False
        for span in entity_rule_spans:
            if start_pos <= span[1] and end_pos >= span[0]:
                overlap_with_rule_span = True
                break
        return overlap_with_rule_span

    for rule in entity_rules:
        for m in re.finditer(rule, text):
            if not is_overlap_with_rule_span(m.start(), m.end()-1):
                entity_rule_spans.append((m.start(), m.end()-1))

    # get entities by pos segger
    word_start_pos = 0
    entity_spans = []
    for word_info in dt.cut(text):
        word_end_pos = word_start_pos + len(word_info.word) - 1
        if word_info.flag in ['s', 't', 'nr', 'ns', 'nt', 'nw', 'nz', 'm', 'r']:
            if not is_overlap_with_rule_span(word_start_pos, word_end_pos):
                entity_spans.append((word_start_pos, word_end_pos))
        word_start_pos = word_end_pos + 1

    # combine rule spans and word spans
    for entity_rule_span in entity_rule_spans:
        for idx, entity_span in enumerate(entity_spans):
            if entity_rule_span[0] < entity_span[0]:
                entity_spans.insert(idx, entity_rule_span)
                break

    return entity_spans


def read_examples(full_file, max_seq_length=None):

    if type(full_file) is str and os.path.isfile(full_file):
        with open(full_file, 'r', encoding='utf-8') as reader:
            full_data = json.load(reader)
    elif type(full_file) is list:
        full_data = full_file

    if max_seq_length is not None and max_seq_length > 0:
        # Separate the over-long context into multiple replica so that the whole context can be processed with the limit
        # The support fact sentences are not processed, so this should only be used for testing.
        expanded_data = []
        for case in full_data:
            sentence_lengths = []
            for para in case['context']:
                for sent in para[1]:
                    sentence_lengths.append(len(sent))
            context_length = sum(sentence_lengths)
            num_sentence = len(sentence_lengths)
            max_context_length = max_seq_length - 3 - len(case['question'])
            if context_length > max_context_length:
                max_shift_sent = int(max_context_length*num_sentence/context_length/2)
                new_context_start_sent = 0
                while new_context_start_sent < num_sentence:
                    new_context_length = 0
                    sent_idx = 0
                    paragraphs = []
                    reach_limit = False
                    for para in case['context']:
                        para_sents = []
                        for sent in para[1]:
                            if sent_idx >= new_context_start_sent:
                                para_sents.append(sent)
                                new_context_length += len(sent)
                                if new_context_length >= max_context_length:
                                    reach_limit = True
                                    break
                            sent_idx += 1
                        if len(para_sents) > 0:
                            paragraphs.append([para[0], para_sents])
                        if reach_limit:
                            break
                    new_case = case.copy()
                    if type(case['_id']) is int:
                        new_case['_id'] = 'INT(%d)' % case['_id']
                    new_case['_id'] += '_SHIFT%d' % new_context_start_sent
                    new_case['context'] = paragraphs
                    expanded_data.append(new_case)

                    if sent_idx >= num_sentence - 1:
                        # reached the end
                        break
                    if sum(sentence_lengths[new_context_start_sent+max_shift_sent:]) >= max_context_length:
                        new_context_start_sent = new_context_start_sent + max_shift_sent
                    else:
                        new_context_start_sent = num_sentence - 1 - np.max(
                            np.flatnonzero(np.cumsum(sentence_lengths[::-1]) <= max_context_length))
            else:
                expanded_data.append(case)
        full_data = expanded_data

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    for case in tqdm(full_data):   
        key = case['_id']
        qas_type = "" # case['type']
        supporting_facts = case.get('supporting_facts', [])
        sup_facts = set([(sp[0], sp[1]) for sp in supporting_facts])
        sup_titles = set([sp[0] for sp in supporting_facts])
        orig_answer_text = case.get('answer', '')

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text == 'unknown' or orig_answer_text=="" # judge_flag??
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        question_entity_spans = get_entity_spans(case['question'])

        for question_entity_span in question_entity_spans:
            # (sentence index, entity start position, entity end position)
            # the position is ralative to the start of the query
            entity_start_end_position.append((-1, question_entity_span[0], question_entity_span[1]))

        # for debug
        titles = set()
        para_data = case['context']
        for paragraph in para_data:  
            title = paragraph[0]
            sents = paragraph[1]   

            titles.add(title)  
            is_gold_para = 1 if title in sup_titles else 0  

            para_start_position = len(doc_tokens)  

            para_entity_spans = get_entity_spans(''.join(sents))
            para_entity_idx = 0

            for local_sent_id, sent in enumerate(sents):  
                if local_sent_id >= 100:  
                    break

                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   
                sent_names.append(local_sent_name)  
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   
                sent_id += 1   
                sent=" ".join(sent)
                sent += " "

                sent_start_word_id = len(doc_tokens)           
                sent_start_char_id = len(char_to_word_offset)  

                for c in sent:  
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1  
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))  

                while (para_entity_idx < len(para_entity_spans) and
                       para_start_position + para_entity_spans[para_entity_idx][0] >= sent_start_word_id and
                       para_start_position + para_entity_spans[para_entity_idx][1] <= sent_end_word_id):
                    # (sentence index, entity start position, entity end position)
                    # the position is relative the start of whole document (all paragraphs)
                    entity_start_end_position.append((len(sent_start_end_position)-1,
                                                      para_start_position + para_entity_spans[para_entity_idx][0],
                                                      para_start_position + para_entity_spans[para_entity_idx][1]))
                    para_entity_idx += 1

                # Answer char position
                answer_offsets = []
                offset = -1

                tmp_answer = " ".join(orig_answer_text)
                while True:

                    offset = sent.find(tmp_answer, offset + 1)
                    if offset != -1:
                        answer_offsets.append(offset)   
                    else:
                        break

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                # only set answer start/end position when answer text in a support fact sentence
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0 and local_sent_name in sup_facts:
                    FIND_FLAG = True   
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   
                        end_char_position = start_char_position + len(tmp_answer) - 1  
                       
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])

                # # Don't do truncation here, the original context/answer/support fact should always be kept in
                # # examples. The truncation is done in 'convert_examples_to_features'
                # if len(doc_tokens) > 382:
                #     break

            if not JUDGE_FLAG and not FIND_FLAG:
                print('no answer found for case {}'.format(key))

            if len(sup_facts_sent_id) < len(sup_facts):
                print('lossing support facts for case {}'.format(key))

            para_end_position = len(doc_tokens) - 1
            
            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))  

        if len(ans_end_position) > 1:
            cnt += 1    
        if type(key) is int and key < 0:
            print("qid {}".format(key))
            print("qas type {}".format(qas_type))
            print("doc tokens {}".format(doc_tokens))
            print("question {}".format(case['question']))
            print("sent num {}".format(sent_id+1))
            print("sup face id {}".format(sup_facts_sent_id))
            print("para_start_end_position {}".format(para_start_end_position))
            print("sent_start_end_position {}".format(sent_start_end_position))
            print("entity_start_end_position {}".format(entity_start_end_position))
            print("orig_answer_text {}".format(orig_answer_text))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        # # print context and questions
        # print('{}: {}|{}$'.format(key, case['question'], ''.join(case['context'][0][1])))
       
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position, 
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,   
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    # max_query_length = 50
    vocab = list(tokenizer.vocab.keys())
    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):  
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0   # 统计answer type

        query_tokens = ["[CLS]"]
        # always use character-level tokenization for question
        question_tokens = list(example.question_text)
        if len(question_tokens) > max_query_length - 2:
            question_tokens = question_tokens[:max_query_length - 2]
        question_tokens = [tok if tok in vocab else '[UNK]' for tok in question_tokens]
        query_tokens.extend(question_tokens)
        query_tokens.append("[SEP]")
        query_to_tok_index = list(range(len(question_tokens)+1))[1:]

        # para_spans = []
        entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        all_doc_tokens = query_tokens.copy()
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))  
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)    
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)  

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            if orig_start_position is None:
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:  
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1  
            
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):  
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)  
            ans_end_position.append(e_pos)

        for sent_span in example.sent_start_end_position:   
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:  # skip single char sentence?
                continue  
            sent_start_position = orig_to_tok_index[sent_span[0]] 
            sent_end_position = orig_to_tok_back_index[sent_span[1]] 
            sentence_spans.append((sent_start_position, sent_end_position)) 

        for entity_span in example.entity_start_end_position:
            if entity_span[0] == -1:
                # entities in query
                if entity_span[2] >= len(query_to_tok_index):
                    continue
                entity_start_position = query_to_tok_index[entity_span[1]]
                entity_end_position = query_to_tok_index[entity_span[2]]
            else:
                # entities in context
                if entity_span[2] >= len(orig_to_tok_index):
                    continue
                entity_start_position = orig_to_tok_index[entity_span[1]]
                entity_end_position = orig_to_tok_back_index[entity_span[2]]
            entity_spans.append((entity_span[0], entity_start_position, entity_end_position))

        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        sentence_spans = get_valid_sentence_spans(sentence_spans, max_seq_length - 1)
        entity_spans = get_valid_entity_spans(entity_spans, len(sentence_spans), max_seq_length - 1)

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1

        if ans_type == 0 and ''.join(all_doc_tokens).find(example.orig_answer_text) < 0:
            # convert 'span' to 'unknown' in case the answer span is missing
            # Note that the condition 'len(ans_start_position) == 0' is not sufficient for answer check since there may
            # be labeling errors (on support facts).
            ans_type = 3
            ans_start_position = []
            ans_end_position = []
            sup_fact_ids = []
        elif ans_type in [1, 2] and len(sup_fact_ids) != len(example.sup_fact_id):
            # convert 'yes'/'no' to 'unknown' in case the support fact is truncated
            # Note that sup_fact_ids is not reset. It's left to model configuration to decide how to use this imcomplete
            # support fact labels.
            # For 'span' answers, no special processing for support fact truncation.
            ans_type = 3

        if type(example.qas_id) is int and example.qas_id < 0:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            print("sentence_spans {}".format(sentence_spans))
            print("entity_spans {}".format(entity_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans+entity_spans,  # append entity span to sentence span list
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_sentence_spans(spans, limit):
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def get_valid_entity_spans(spans, sentence_limit, sequence_limit):
    new_spans = [span for span in spans if span[0] < sentence_limit and span[2] < sequence_limit]
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../libs'))
    from bert import tokenization

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from reading_comprehension.separate_train_data import save_to_folder

    parser = argparse.ArgumentParser()
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    # Other parameters
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")
    parser.add_argument("--full_data", type=str, required=True)   
    parser.add_argument('--tokenizer_path', type=str, required=True)

    args = parser.parse_args()
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(args.tokenizer_path, 'vocab.txt'),
                                           do_lower_case=True)

    # examples = read_examples(full_file=args.full_data, max_seq_length=args.max_seq_length)

    if os.path.isfile(args.example_output) and args.example_output.endswith('.pkl.gz'):
        with gzip.open(args.example_output, 'rb') as f:
            examples = pickle.load(f)
    else:
        examples = read_examples(full_file=args.full_data)
        if args.example_output.endswith('.pkl.gz'):
            with gzip.open(args.example_output, 'wb') as fout:
                pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=args.max_seq_length,
                                            max_query_length=50)

    if args.feature_output.endswith('.pkl.gz'):
        with gzip.open(args.feature_output, 'wb') as fout:
            pickle.dump(features, fout)
    else:
        if not args.example_output.endswith('.pkl.gz'):
            save_to_folder(features, args.feature_output, examples, args.example_output)
        else:
            save_to_folder(features, args.feature_output)
