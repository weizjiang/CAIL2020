"""
prepare data for training/testing
"""
import os
import json
import random
import re
import numpy as np


def separate_dev_set():
    # input_file = r'C:\Works\DataSet\CAIL\ydlj_small_data\train.json'
    # train_data_file = r'data\train_small.json'
    # dev_data_file = r'data\dev_small.json'
    # train_set_ratio = 0.8

    input_file = r'C:\Works\DataSet\CAIL\ydlj_big_data\train.json'
    train_data_file = r'data\train_big.json'
    dev_data_file = r'data\dev_big.json'
    train_set_ratio = 0.9

    with open(input_file, 'r', encoding='utf-8') as f_in:
        all_data = json.load(f_in)
    
    random.shuffle(all_data)
    train_set_size = int(len(all_data) * train_set_ratio)
    
    with open(train_data_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_data[:train_set_size], f_out, ensure_ascii=False, indent=4)
    
    with open(dev_data_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_data[train_set_size:], f_out, ensure_ascii=False, indent=4)


def combine_support_facts():
    # combine the support facts so that the answer only appears in one sentence

    input_file = r'data\train_big.json'
    output_file = r'data\train_big_combine_answer.json'

    # input_file = r'data\dev_big.json'
    # output_file = r'data\dev_big_combine_answer.json'

    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    for item in data:
        if item["answer"] in ['yes', 'no', 'unknown']:
            continue

        find_answer = False
        for support_fact in item["supporting_facts"]:
            if item['context'][0][1][support_fact[1]].find(item["answer"]) >= 0:
                find_answer = True
                break

        if not find_answer:
            context = item['context'][0][1]
            combined_support_fact_text = ''
            support_fact_starts = []
            for support_fact in item["supporting_facts"]:
                support_fact_text = context[support_fact[1]]
                support_fact_starts.append(len(combined_support_fact_text))
                combined_support_fact_text += support_fact_text
            ans_pos = combined_support_fact_text.find(item["answer"])
            if ans_pos >= 0:
                support_fact_ends = np.array(support_fact_starts[1:] + [len(combined_support_fact_text)])
                support_fact_starts = np.array(support_fact_starts)
                support_fact_indices = np.flatnonzero(
                    np.logical_and(support_fact_ends > ans_pos, support_fact_starts < ans_pos + len(item["answer"])))

                support_fact_with_answer = [support_fact[1] for idx, support_fact in enumerate(item["supporting_facts"])
                                            if idx in support_fact_indices]
                combined_support_fact = ''.join([context[idx] for idx in support_fact_with_answer])
                context[support_fact_with_answer[0]] = combined_support_fact
                context = [sent for idx, sent in enumerate(context) if idx not in support_fact_with_answer[1:]]
                item['context'][0][1] = context

                reduced_support_facts = []
                for support_fact in item["supporting_facts"]:
                    if support_fact[1] <= support_fact_with_answer[0]:
                        reduced_support_facts.append(support_fact)
                    elif support_fact[1] > support_fact_with_answer[-1]:
                        support_fact[1] = support_fact[1] - len(support_fact_with_answer) + 1
                        reduced_support_facts.append(support_fact)
                item["supporting_facts"] = reduced_support_facts

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)


def combine_data_set():
    # input_file_1 = r'data\train_small.json'
    # input_file_2 = r'data\train_big.json'
    # output_file = r'data\train_2020.json'

    # input_file_1 = r'data\dev_small.json'
    # input_file_2 = r'data\dev_big.json'
    # output_file = r'data\dev_2020.json'

    # id_prefix_1 = 'cail2020_small_'
    # id_prefix_2 = 'cail2020_big_'

    # input_file_1 = r'data\data2020_combine2019all_cmrc2018all_1sentence_augmented\train.json'
    # # input_file_2 = r'data\all_2019_1sentence_converted_augmented.json'
    # input_file_2 = r'data\cmrc2018_all_1sentence_converted_augmented.json'
    # output_file = input_file_1
    #
    # id_prefix_1 = ''
    # id_prefix_2 = ''

    # input_file_1 = r'data\data_big_combine2019all_cmrc2018all_1sentence\train.json'
    # # input_file_2 = r'data\all_2019_1sentence_converted.json'
    # input_file_2 = r'data\cmrc2018_all_1sentence_converted.json'
    # output_file = input_file_1

    input_file_1 = r'data\data_big_combine2019all_cmrc2018all_1sentence_augmented\train.json'
    # input_file_2 = r'data\all_2019_1sentence_converted_augmented.json'
    input_file_2 = r'data\cmrc2018_all_1sentence_converted_augmented.json'
    output_file = input_file_1

    # id_prefix_1 = 'cail2020_'
    id_prefix_1 = ''
    id_prefix_2 = ''

    with open(input_file_1, 'r', encoding='utf-8') as f_in:
        data_1 = json.load(f_in)
        if len(id_prefix_1) > 0:
            for item in data_1:
                item['_id'] = '{}{}'.format(id_prefix_1, item['_id'])

    with open(input_file_2, 'r', encoding='utf-8') as f_in:
        data_2 = json.load(f_in)
        if len(id_prefix_2) > 0:
            for item in data_2:
                item['_id'] = '{}{}'.format(id_prefix_2, item['_id'])

    data = data_1 + data_2
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)


def generate_dev_result():
    # dev_data_file = r'data\dev_small.json'
    # dev_result_file = r'data\dev_small_result.json'
    dev_data_file = r'data\dev_big.json'
    dev_result_file = r'data\dev_big_result.json'

    with open(dev_data_file, 'r', encoding='utf-8') as f_in:
        dev_date = json.load(f_in)

    dev_result = {
        "answer": {},
        "sp": {}
    }

    for item in dev_date:
        dev_result["answer"][item["_id"]] = item["answer"]
        dev_result["sp"][item["_id"]] = item["supporting_facts"]

    with open(dev_result_file, 'w', encoding='utf-8') as f_out:
        json.dump(dev_result, f_out, ensure_ascii=False, indent=4)


def convert_cail2019_data(dataset='train', separate_paragraph=False):
    if dataset == 'train':
        input_file = r'C:\Works\DataSet\CAIL\CAIL2019\big_train_data.json'
        output_file = r'data\train_2019.json'
        if separate_paragraph:
            converted_file = r'data\train_2019_converted.json'
        else:
            converted_file = r'data\train_2019_1sentence_converted.json'
        id_prefix = ''
    elif dataset == 'dev':
        input_file = r'C:\Works\DataSet\CAIL\CAIL2019\dev_ground_truth.json'
        output_file = r'data\dev_2019.json'
        if separate_paragraph:
            converted_file = r'data\dev_2019_converted.json'
        else:
            converted_file = r'data\dev_2019_1sentence_converted.json'
        id_prefix = 'cail2019_dev_'
    elif dataset == 'test':
        input_file = r'C:\Works\DataSet\CAIL\CAIL2019\test_ground_truth.json'
        output_file = r'data\test_2019.json'
        if separate_paragraph:
            converted_file = r'data\test_2019_converted.json'
        else:
            converted_file = r'data\test_2019_1sentence_converted.json'
        id_prefix = 'cail2019_test_'

    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    # with open(output_file, 'w', encoding='utf-8') as f_out:
    #     json.dump(data, f_out, ensure_ascii=False, indent=4)

    converted_data = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context_text = clean_text(paragraph['context'])
            sentences, sentence_spans = separate_sentence(context_text)
            for qa in paragraph['qas']:
                if qa['is_impossible'] == 'true':
                    answer_text = 'unknown'
                    context = [paragraph['casename'], sentences if separate_paragraph else [context_text]]
                    supporting_facts = []
                else:
                    if len(qa['answers']) == 1:
                        # only one answer in train set
                        answer = qa['answers'][0]
                    else:
                        # three answers in dev/test set
                        answer_list = [clean_text(answer_option['text']) for answer_option in qa['answers']]
                        answer_set = set(answer_list)
                        if len(answer_set) == 1:
                            # all same
                            answer = qa['answers'][0]
                        elif len(answer_set) == len(answer_list):
                            # use the shortest one if all different
                            answer_len = [len(a) for a in answer_list]
                            answer_idx = np.argmin(answer_len)
                            answer = qa['answers'][answer_idx]
                        else:
                            # use the most voted one
                            for answer_idx, answer_text in enumerate(answer_list):
                                if answer_list.count(answer_text) > 1:
                                    answer = qa['answers'][answer_idx]
                                    break

                    if answer['answer_start'] == -1:
                        # 'YES'/'NO' answer
                        answer_text = answer['text'].lower()
                        # use whole paragrpah as a sentence, since no support fact info avaliable
                        context = [paragraph['casename'], [context_text]]
                        supporting_facts = [[paragraph['casename'], 0]]
                    else:
                        # span answer
                        answer_text = clean_text(answer['text'])

                        # due to cleaning, need to re-find the answer text pos
                        # it should be close to the original position
                        # --> find all occurance as support fact
                        answer_positions = []
                        while True:
                            if len(answer_positions) > 0:
                                search_start = answer_positions[-1] + len(answer_text)
                            else:
                                search_start = 0
                            # search_end = min(answer['answer_start'] + len(answer_text), len(context_text))
                            # find_pos = context_text[search_start:search_end].find(answer_text)
                            # --> search whole context since the original 'answer_start' is only the first occurance
                            find_pos = context_text[search_start:].find(answer_text)
                            if find_pos >= 0:
                                answer_start = search_start + find_pos
                                answer_positions.append(answer_start)
                            else:
                                break
                        assert len(answer_positions) > 0

                        if separate_paragraph:
                            support_fact_indices = set()
                            for answer_start in answer_positions:
                                # allow an answer span multiple sentences?
                                sentence_idx = np.flatnonzero(
                                    np.logical_and(sentence_spans[:, 1] > answer_start,
                                                   sentence_spans[:, 0] < answer_start + len(answer_text))
                                )
                                support_fact_indices = support_fact_indices.union(set(sentence_idx))

                            if len(support_fact_indices) > 0:
                                # only use sentences that answer span appears, it's inaccurate!
                                context = [paragraph['casename'], sentences]
                                supporting_facts = [[paragraph['casename'], int(idx)] for idx in support_fact_indices]
                            else:
                                context = [paragraph['casename'], [context_text]]
                                supporting_facts = [[paragraph['casename'], 0]]
                        else:
                            context = [paragraph['casename'], [context_text]]
                            supporting_facts = [[paragraph['casename'], 0]]

                qa_item = {
                    '_id': id_prefix + qa['id'],
                    'context': [context],
                    'question': qa['question'],
                    'answer': answer_text,
                    # 'answer_start' in cail2019 is always the first occurence in the context, not imformative
                    # 'answer_start': answer_start,
                    'supporting_facts': supporting_facts
                }

                converted_data.append(qa_item)

    with open(converted_file, 'w', encoding='utf-8') as f_out:
        json.dump(converted_data, f_out, ensure_ascii=False, indent=4)


def clean_text(text):
    text = re.sub(r'\&nbsp;?', ' ', text)
    text = re.sub(r'\&ensp;?', ' ', text)
    text = re.sub(r'\&amp;?', '&', text)
    text = re.sub(r'\&rarr;?', '→', text)
    text = re.sub(r'\&ldquo;?', '“', text)
    text = re.sub(r'\&rdquo;?', '”', text)
    text = re.sub(r'\&lsquo;?', '‘', text)
    text = re.sub(r'\&rsquo;?', '’', text)
    text = re.sub(r'\&ndash;?', '–', text)
    text = re.sub(r'\&mdash;?', '—', text)
    text = re.sub(r'\&hellip;?', '…', text)
    text = re.sub(r'\&times;?', '×', text)
    text = re.sub(r'\&divide;?', '÷', text)
    text = re.sub(r'\&permil;?', '‰', text)
    text = re.sub(r'\&asymp;?', '≈', text)
    text = re.sub(r'\&lt;?', '<', text)
    text = re.sub(r'\&gt;?', '>', text)
    text = re.sub(r'\&middot;?', '·', text)
    text = re.sub(r'\&bull;?', '•', text)
    text = re.sub(r'\&Omicron;?', '〇', text)
    text = re.sub(r'\&$', '', text)

    return text


def separate_sentence(text, separators=None, non_starting_chars=None):
    """ Separate a sentence by any separator listed in separators.
    A separator can be a group of charactors.
    A non-starting charactor following a separator forbids the separation."""
    if separators is None:
        separators = ['。', '？', '！', '；', r'\?', '!', ';', ',', '，']

    if non_starting_chars is None:
        non_starting_chars = ['，', '；', '：', '、', '。', '？', '！', '”', '’', '）', '」', '》', '】', '』', '〉',
                              ',', ';', ':', '!', r'\?', r'\]', r'\}', r'\)']

    separated_text = re.sub(r'({0}\s*)(?![{1}])'.format('|'.join(separators), ''.join(non_starting_chars)), r'\1\n', text)

    sentences = [sentence for sentence in separated_text.split('\n') if len(sentence) > 0]
    lengths = [len(sentence) for sentence in sentences]
    assert (sum(lengths) == len(text))
    accumulated_lengths = list(np.cumsum(lengths))
    sentence_spans = np.vstack([[0]+accumulated_lengths[:-1], accumulated_lengths]).T

    return sentences, sentence_spans


def convert_cmrc2018_data(dataset='train', separate_paragraph=False):
    id_prefix = 'CMRC2018_'
    if dataset == 'train':
        input_file = r'C:\Works\DataSet\CMRC2018\cmrc2018_train.json'
        output_file = r'data\cmrc2018_train.json'
        if separate_paragraph:
            converted_file = r'data\cmrc2018_train_converted.json'
        else:
            converted_file = r'data\cmrc2018_train_1sentence_converted.json'
    elif dataset == 'dev':
        input_file = r'C:\Works\DataSet\CMRC2018\cmrc2018_dev.json'
        output_file = r'data\cmrc2018_dev.json'
        if separate_paragraph:
            converted_file = r'data\cmrc2018_dev_converted.json'
        else:
            converted_file = r'data\cmrc2018_dev_1sentence_converted.json'
    elif dataset == 'trial':
        input_file = r'C:\Works\DataSet\CMRC2018\cmrc2018_trial.json'
        output_file = r'data\cmrc2018_trial.json'
        if separate_paragraph:
            converted_file = r'data\cmrc2018_trial_converted.json'
        else:
            converted_file = r'data\cmrc2018_trial_1sentence_converted.json'

    if os.path.isfile(output_file):
        with open(output_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    else:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=4)

    converted_data = []
    for item in data:
        context_text = item['context_text'].replace('\n', '')
        for qa in item['qas']:
            if len(qa['answers']) == 1:
                # only one answer in train set
                answer = str(qa['answers'][0])
            else:
                # three answers in dev set
                answer_list = [str(a) for a in qa['answers']]
                answer_set = set(answer_list)
                if len(answer_set) == 1:
                    # all same
                    answer = answer_list[0]
                elif len(answer_set) == len(answer_list):
                    # use the shortest one if all different
                    answer_len = [len(a) for a in answer_list]
                    answer_idx = np.argmin(answer_len)
                    answer = answer_list[answer_idx]
                else:
                    # use the most voted one
                    for answer_idx, answer_text in enumerate(answer_list):
                        if answer_list.count(answer_text) > 1:
                            answer = answer_list[answer_idx]
                            break

            # span answer
            answer_text = answer

            while True:
                # find all occurance as support fact
                answer_positions = []
                while True:
                    if len(answer_positions) > 0:
                        search_start = answer_positions[-1] + len(answer_text)
                    else:
                        search_start = 0
                    # search whole context since the original 'answer_start' is only the first occurance
                    find_pos = context_text[search_start:].find(answer_text)
                    if find_pos >= 0:
                        answer_start = search_start + find_pos
                        answer_positions.append(answer_start)
                    else:
                        break

                if len(answer_positions) == 0:
                    # the original data may have probelm of answer mismatch
                    if answer_text[-1] in ['，', '。', '；']:
                        answer_text = answer_text[:-1]
                else:
                    break

            assert len(answer_positions) > 0
            assert len(answer_text) > 0

            if separate_paragraph:
                sentences, sentence_spans = separate_sentence(context_text)
                support_fact_indices = set()
                for answer_start in answer_positions:
                    # allow an answer span multiple sentences?
                    sentence_idx = np.flatnonzero(
                        np.logical_and(sentence_spans[:, 1] > answer_start,
                                       sentence_spans[:, 0] < answer_start + len(answer_text))
                    )
                    support_fact_indices = support_fact_indices.union(set(sentence_idx))

                if len(support_fact_indices) > 0:
                    # only use sentences that answer span appears, it's inaccurate!
                    context = [item['title'], sentences]
                    supporting_facts = [[item['title'], int(idx)] for idx in support_fact_indices]
                else:
                    context = [item['title'], [context_text]]
                    supporting_facts = [[item['title'], 0]]
            else:
                context = [item['title'], [context_text]]
                supporting_facts = [[item['title'], 0]]

            qa_item = {
                '_id': id_prefix + qa['query_id'],
                'context': [context],
                'question': qa['query_text'],
                'answer': answer_text,
                'supporting_facts': supporting_facts
            }

            converted_data.append(qa_item)

    with open(converted_file, 'w', encoding='utf-8') as f_out:
        json.dump(converted_data, f_out, ensure_ascii=False, indent=4)

def generate_test_file():
    data_file = r'data/dev.json'
    with open(data_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    for item in data:
        item.pop("answer")
        item.pop("supporting_facts")

    test_file = "./input/data.json"
    with open(test_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)


def analyze_data():
    # data_file = r'data/train.json'
    # data_file = r'data/dev.json'
    data_file = r'data/train_2019_1sentence_converted.json'
    # data_file = r'data/data_combine2019_1sentence/train.json'

    with open(data_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    answer_types = ['span', 'yes', 'no', 'unknown']
    answers = [item['answer'] if item['answer'] in answer_types[1:] else 'span' for item in data]
    span_lengths = [len(item['answer']) for item in data if item['answer'] not in answer_types[1:]]
    print('-- ', data_file)
    print('total samples: {}'.format(len(answers)))
    for answer_type in answer_types:
        type_count = answers.count(answer_type)
        print('{}: {:.1f}% ({})'.format(answer_type, 100 * type_count/len(answers), type_count))

    print('max span length: {}\n'.format(max(span_lengths)))

    print('span length > 50:')
    num_long_span = 0
    for item in data:
        if len(item['answer']) >= 50:
            num_long_span += 1
            print(item['answer'])
    print('span length > 50 #sample: {} in {}\n'.format(num_long_span, len(data)))

    print('answer with punctuation:')
    for item in data:
        if item['answer'].find('，') >= 0 or item['answer'].find('。') >= 0 or item['answer'].find('；') >= 0:
            print(item['answer'])
            # print(item)

    num_sentences = [len(item['context'][0][1]) for item in data]
    print('\nmax number of sentences: {}'.format(max(num_sentences)))


def augment_data_single_hop(num_delete=3, num_shuffle=3):
    """
    do augmentation by deleting ans shuffling the sentences, generate new samples: num_delete x num_shuffle
    the augmented data is saved as single sentence context format
    :param num_delete:
    :param num_shuffle:
    :return:
    """
    # in_file = r'data/all_2019_converted.json'
    # out_file = r'data/all_2019_1sentence_converted_augmented.json'
    in_file = r'data/cmrc2018_all_converted.json'
    out_file = r'data/cmrc2018_all_1sentence_converted_augmented.json'

    with open(in_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    augmented_data = []
    for qa in data:
        context_sentences = qa["context"][0][1]
        num_sentence = len(context_sentences)
        support_fact_indices = [support_fact[1] for support_fact in qa["supporting_facts"]]

        # add the original context, converting to single sentence
        qa.update({
            'context': [[qa["context"][0][0], [''.join(context_sentences)]]],
            'supporting_facts': [[qa["context"][0][0], 0]] if len(support_fact_indices) > 0 else []
        })
        augmented_data.append(qa)

        if len(support_fact_indices) > 0:
            # reserve some sentences ahead/behind
            support_fact_start = max(min(support_fact_indices) - 2, 0)
            support_fact_end = min(max(support_fact_indices) + 2, num_sentence-1)
            non_support_facts = list(range(0, support_fact_start)) + list(range(support_fact_end+1, num_sentence))
        else:
            support_fact_start = None
            support_fact_end = None
            non_support_facts = list(range(0, num_sentence))

        if len(non_support_facts) == 0:
            continue

        for del_idx in range(num_delete):
            sent_list = non_support_facts.copy()
            max_num_del = max(1, int(len(sent_list) * 0.5))
            num_del = np.random.randint(1, max_num_del + 1)
            rand_positions = np.random.choice(len(sent_list), num_del, replace=False)
            indices = np.delete(sent_list, rand_positions)
            if len(support_fact_indices) > 0:
                # occupy a position for support facts
                post_positions = np.flatnonzero(indices > support_fact_start)
                indices = np.insert(indices, post_positions[0] if len(post_positions) > 0 else len(indices), -1)
            for shuffle_idx in range(num_shuffle):
                max_num_shift = max(1, int(len(indices) * 0.5))
                num_shift = np.random.randint(1, max_num_shift + 1)
                rand_positions = np.random.choice(len(indices), num_shift, replace=False)
                permutate_pos = np.random.permutation(rand_positions)
                indices[rand_positions] = indices[permutate_pos]

                # combine the sentences
                sentence_list = []
                for idx in indices:
                    if idx == -1:
                        sentence_list += context_sentences[support_fact_start:support_fact_end]
                    else:
                        sentence_list.append(context_sentences[idx])

                augmented_qa = qa.copy()
                augmented_qa.update({
                    '_id': '%s_%d_%d' % (qa['_id'], del_idx, shuffle_idx),
                    'context': [[qa["context"][0][0], [''.join(sentence_list)]]],
                    'supporting_facts': [[qa["context"][0][0], 0]] if len(support_fact_indices) > 0 else []
                })
                augmented_data.append(augmented_qa)

    with open(out_file, 'w', encoding='utf-8') as f_out:
        json.dump(augmented_data, f_out, ensure_ascii=False, indent=4)


def augment_data_multi_hop(num_delete=10, num_shuffle=10):
    """
    do augmentation by deleting ans shuffling the sentences, generate new samples: num_delete x num_shuffle
    :param num_delete:
    :param num_shuffle:
    :return:
    """
    in_file = r'data/train_big_combine_answer.json'
    out_file = r'data/train_big_combine_answer_augmented.json'

    with open(in_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    augmented_data = []
    for qa in data:
        context_sentences = qa["context"][0][1]
        num_sentence = len(context_sentences)
        support_fact_indices = [support_fact[1] for support_fact in qa["supporting_facts"]]

        # add the original context
        augmented_data.append(qa)

        non_support_facts = [idx for idx in range(num_sentence) if idx not in support_fact_indices]

        if len(non_support_facts) == 0:
            continue

        for del_idx in range(num_delete):
            sent_list = list(range(num_sentence))
            max_num_del = max(1, int(len(non_support_facts) * 0.5))
            num_del = np.random.randint(1, max_num_del + 1)
            rand_positions = np.random.choice(non_support_facts, num_del, replace=False)
            indices = np.delete(sent_list, rand_positions)
            for shuffle_idx in range(num_shuffle):
                sub_indices = np.array([i for i, idx in enumerate(indices) if idx not in support_fact_indices])
                max_num_shift = max(1, int(len(sub_indices) * 0.5))
                num_shift = np.random.randint(1, max_num_shift + 1)
                rand_positions = np.random.choice(sub_indices, num_shift, replace=False)
                permutate_pos = np.random.permutation(rand_positions)
                indices[rand_positions] = indices[permutate_pos]

                # combine the sentences
                sentence_list = [context_sentences[idx] for idx in indices]

                new_support_fact_indices = [np.flatnonzero(indices == idx)[0] for idx in support_fact_indices]

                augmented_qa = qa.copy()
                augmented_qa.update({
                    '_id': '%s_%d_%d' % (qa['_id'], del_idx, shuffle_idx),
                    'context': [[qa["context"][0][0], sentence_list]],
                    'supporting_facts': [[qa["context"][0][0], int(idx)] for idx in new_support_fact_indices]
                })
                augmented_data.append(augmented_qa)

    with open(out_file, 'w', encoding='utf-8') as f_out:
        json.dump(augmented_data, f_out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # separate_dev_set()

    # combine_support_facts()

    # combine_data_set()

    generate_dev_result()

    # convert_cail2019_data(dataset='train', separate_paragraph=True)

    # convert_cmrc2018_data(dataset='dev', separate_paragraph=False)

    # augment_data_single_hop()

    # augment_data_multi_hop()

    # generate_test_file()

    # analyze_data()

    # with open(r'data/data_combine2019all_1sentence_augmented/train.json', 'r', encoding='utf-8') as f_in:
    #     data = json.load(f_in)
    # print(len(data))
