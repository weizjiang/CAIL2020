"""
prepare data for training/testing
"""
import json
import random
import re
import numpy as np


def separate_dev_set():
    input_file = r'C:\Works\DataSet\CAIL\ydlj_small_data\train.json'
    train_data_file = r'data\train.json'
    dev_data_file = r'data\dev.json'
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        all_data = json.load(f_in)
    
    random.shuffle(all_data)
    train_set_ratio = 0.8
    train_set_size = int(len(all_data) * train_set_ratio)
    
    with open(train_data_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_data[:train_set_size], f_out, ensure_ascii=False, indent=4)
    
    with open(dev_data_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_data[train_set_size:], f_out, ensure_ascii=False, indent=4)


def generate_dev_result():
    dev_data_file = r'data\dev.json'
    dev_result_file = r'data\dev_result.json'

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


def convert_cail2019_data():
    input_file = r'C:\Works\DataSet\CAIL\CAIL2019\big_train_data.json'
    output_file = r'data\train_2019.json'
    converted_file = r'data\train_2019_converted.json'

    # input_file = r'C:\Works\DataSet\CAIL\CAIL2019\dev_ground_truth.json'
    # output_file = r'data\dev_2019.json'

    # input_file = r'C:\Works\DataSet\CAIL\CAIL2019\test_ground_truth.json'
    # output_file = r'data\test_2019.json'

    with open(input_file, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    # with open(output_file, 'w', encoding='utf-8') as f_out:
    #     json.dump(data, f_out, ensure_ascii=False, indent=4)

    converted_data = []
    for item in data['data']:
        for paragrph in item['paragraphs']:
            context_text = clean_text(paragrph['context'])
            sentences, sentence_spans = separate_sentence(context_text)
            for qa in paragrph['qas']:
                if qa['is_impossible'] == 'true':
                    answer_text = 'unknown'
                    context = [paragrph['casename'], sentences]
                    supporting_facts = []
                else:
                    # only one answer in train set
                    answer = qa['answers'][0]

                    if answer['answer_start'] == -1:
                        # 'YES'/'NO' answer
                        answer_text = answer['text'].lower()
                        # use whole paragrpah as a sentence, since no support fact info avaliable
                        context = [paragrph['casename'], [context_text]]
                        supporting_facts = [[paragrph['casename'], 0]]
                    else:
                        # span answer
                        answer_text = clean_text(answer['text'])
                        # due to cleaning, need to re-find the answer text pos
                        # it should be close to the original position
                        answer_start = -1
                        while True:
                            if answer_start == -1:
                                search_start = 0
                            else:
                                search_start = answer_start + len(answer_text)
                            search_end = min(answer['answer_start'] + len(answer_text), len(context_text))
                            find_pos = context_text[search_start:search_end].find(answer_text)
                            if find_pos >= 0:
                                answer_start = search_start + find_pos
                            else:
                                break
                        assert(answer_start >= 0)

                        sentence_idx = np.flatnonzero(
                            np.logical_and(sentence_spans[:, 1] > answer_start,
                                           sentence_spans[:, 0] < answer_start + len(answer_text))
                        )
                        if len(sentence_idx) == 1:
                            # only use one sentence as the supporting fact, may be incomplete!
                            context = [paragrph['casename'], sentences]
                            supporting_facts = [[paragrph['casename'], int(sentence_idx[0])]]
                        elif len(sentence_idx) > 1:
                            # allow an answer span multiple sentences?
                            context = [paragrph['casename'], sentences]
                            supporting_facts = [[paragrph['casename'], int(idx)] for idx in sentence_idx]
                        else:
                            context = [paragrph['casename'], [context_text]]
                            supporting_facts = [[paragrph['casename'], 0]]

                qa_item = {
                    '_id': qa['id'],
                    'context': [context],
                    'question': qa['question'],
                    'answer': answer_text,
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


if __name__ == '__main__':
    # separate_dev_set()

    # generate_dev_result()

    convert_cail2019_data()
