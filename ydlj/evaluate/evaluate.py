import sys
import ujson as json
import re
import string
from collections import Counter
import pickle


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = list(normalized_prediction)
    ground_truth_tokens = list(normalized_ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def eval(prediction_file, gold_file):
    with open(prediction_file, 'r', encoding='utf-8') as f:
        prediction = json.load(f)
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = str(dp['_id'])
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    # eval(sys.argv[1], sys.argv[2])

    # prediction_file = r'../data/dev_result.json'
    # prediction_file = r'../result/train_v1_pred_seed_62_epoch_10_99999.json'
    # prediction_file = r'../result/train_v2_pred_seed_5_epoch_9_2706.json'
    # prediction_file = r'../result/train_v2_pred_seed_5_epoch_8_99999.json'
    # prediction_file = r'../result/train_v3_pred_seed_74_epoch_10_99999.json'
    # prediction_file = r'../result/train_v4_pred_seed_82_epoch_10_99999.json'

    # prediction_file = r'../reading_comprehension/results/result_rc_20200701-211818_threshold_0.5.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200701-211818_threshold_0.2.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200702-144524_threshold_0.2.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200702-144524_threshold_0.1.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-193921_threshold_0.1.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-193921_threshold_0.2.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-193921_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-204721_threshold_0.2.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-204721_threshold_0.5.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200707-204721_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200708-211940_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200708-212137_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200709-160717_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200709-163056_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200709-163056_threshold_0.2_restrict_span_L.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200710-120216_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200710-120452_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200713-131312_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200713-151448_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200714-202143_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200715-172519_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200716-110212_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200716-102909_threshold_0.2_restrict_span_B.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200716-102909_threshold_0.2_restrict_span_L.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200716-182107_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200717-165251_threshold_0.2_restrict_span.json'

    # prediction_file = r'../result/result_2.json'
    # prediction_file = r'../result/result_adarc.json'

    # gold_file = r'../data/dev_small.json'

    # prediction_file = r'../reading_comprehension/results/result_rc_20200720-204118_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200720-205600_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200721-104351_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200722-122420_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200722-152557_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200723-135847_threshold_0.2_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200723-135847_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200723-211344_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200726-185728_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200727-110232_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200727-112404_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200727-204614_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200727-210204_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200728-160636_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200729-121944_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200729-115544_threshold_0.5_restrict_span.json'
    # prediction_file = r'../reading_comprehension/results/result_rc_20200730-210101_threshold_0.5_restrict_span.json'
    prediction_file = r'../reading_comprehension/results/result_rc_20200731-101153_threshold_0.5_restrict_span.json'

    gold_file = r'../data/dev_big.json'

    eval(prediction_file, gold_file)
