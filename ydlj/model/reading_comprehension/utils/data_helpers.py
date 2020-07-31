import json
from collections import OrderedDict
from pylab import *
from tflearn.data_utils import pad_sequences
import numpy as np


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_values):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted scores provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_values: The all predict values by threshold
    Raises:
        IOError: If the prediction file is not a .json file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_values = [round(i, 4) for i in all_predict_values[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('testid', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_values', predict_values)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_label_using_scores_by_threshold(scores, threshold=0.5, topN=3, allow_empty=False):
    """
    Get the predicted labels based on the threshold.
    If there is no predict value greater than threshold, then choose the label which has the max predict value.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
        topN: the max number of labels, -1 for all labels
        allow_empty: if False, the label with the highest score is returned if all lower then threshold
    Returns:
        predicted_labels: The predicted labels
        predicted_values: The predicted values
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for idx, score in enumerate(scores):
        count = 0
        index_list = []
        value_list = []
        for index, predict_value in enumerate(score):
            if predict_value > threshold:
                index_list.append(index)
                value_list.append(predict_value)
                count += 1
        if count == 0 and not allow_empty:
            index_list.append(np.argmax(score))
            value_list.append(max(score))
        elif count > 1:
            # sort in decending order
            sort_order = np.argsort(value_list)[::-1]
            if topN > 0:
                value_list = np.asarray(value_list)[sort_order[0:topN]].tolist()
                index_list = np.asarray(index_list)[sort_order[0:topN]].tolist()
            else:
                value_list = np.asarray(value_list)[sort_order].tolist()
                index_list = np.asarray(index_list)[sort_order].tolist()
        predicted_labels.append(index_list)
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def get_label_using_scores_by_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        value_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            value_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def cal_metric(predicted_labels, labels):
    """
    Calculate the metric(recall, precision).

    Args:
        predicted_labels: The predicted_labels
        labels: The true labels
    Returns:
        The value of metric
    """
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    recall = count / len(label_no_zero)
    precision = count / len(predicted_labels)
    return recall, precision


def cal_metric_batch(predicts, labels):
    numerator = 0
    recall_denominator = 0
    precision_denominator = 0
    accu_cnt = 0
    for predict, label in zip(predicts, labels):
        intersection = set(predict).intersection(set(label))
        union = set(predict).union(set(label))
        numerator += len(intersection)
        recall_denominator += len(label)
        precision_denominator += len(predict)
        if len(intersection) == len(union) and len(intersection) > 0:
            # not considering empty predition in accuracy
            accu_cnt += 1

    if recall_denominator == 0:
        recall = float('nan')
    else:
        recall = numerator / recall_denominator
    if precision_denominator == 0:
        precision = float('nan')
    else:
        precision = numerator / precision_denominator
    accu = accu_cnt/len(predicts)
    f1 = cal_F(recall, precision)
    return accu, recall, precision, f1


def cal_F(recall, precision):
    """
    Calculate the metric F value.

    Args:
        recall: The recall value
        precision: The precision value
    Returns:
        The F value
    """
    F = 0.0
    if np.isnan(recall) or np.isnan(precision):
        F = float('NaN')
    elif recall + precision == 0:
        F = 0.0
    else:
        F = (2 * recall * precision) / (recall + precision)
    return F


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
    return pad_seq, onehot_labels


def plot_seq_len(data_file, data, percentage=0.98):
    """
    Visualizing the sentence length of each data sentence.

    Args:
        data_file: The data_file
        data: The class Data (includes the data tokenindex and data labels)
        percentage: The percentage of the total data you want to show
    """
    data_analysis_dir = '../data/data_analysis/'
    if 'train' in data_file.lower():
        output_file = data_analysis_dir + 'Train Sequence Length Distribution Histogram.png'
    if 'validation' in data_file.lower():
        output_file = data_analysis_dir + 'Validation Sequence Length Distribution Histogram.png'
    if 'test' in data_file.lower():
        output_file = data_analysis_dir + 'Test Sequence Length Distribution Histogram.png'
    result = dict()
    for x in data.tokenindex:
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if count > data.number * percentage:
            border_index.append(item[0])
    avg = avg / data.number
    print('The average of the data sequence length is {0}'.format(avg))
    print('The recommend of padding sequence length should more than {0}'.format(border_index[0]))
    xlim(0, 400)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()

