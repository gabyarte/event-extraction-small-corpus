import pandas as pd
import numpy as np


def lcs(str1, str2):
    n, m = len(str1), len(str2)
    _lcs = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    maximum = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                _lcs[i][j] = 0
            elif str1[i - 1] == str2[j - 1]:
                _lcs[i][j] = _lcs[i - 1][j - 1] + 1
                maximum = max(maximum, _lcs[i][j])
            else:
                _lcs[i][j] = 0
    return maximum


def similarity_meassure(str1, str2):
    max_lcs = lcs(str1, str2)
    return max_lcs / (len(str1) + len(str2) - max_lcs)


def similarity_score(str1, str2):
    if str1 == str2:
        return 1
    if str1 is None or str2 is None:
        return 0
    return similarity_meassure(str1, str2)


def compute_average_similarity_score(true_data, predict_data, label):
    data = []
    for y_true, y_pred in zip(true_data, predict_data):
        if isinstance(y_true[label], list) or isinstance(y_pred[label], list):
            data += similarity_zip(y_true[label], y_pred[label])
        else:
            data += [(y_true[label], y_pred[label])]
    return np.average([
        similarity_score(y_true, y_pred)
        for y_true, y_pred in data
    ])


def compute_similar_observations_by_threshold(true_data, predict_data, label, threshold=0.5):
    data = []
    for y_true, y_pred in zip(true_data, predict_data):
        if isinstance(y_true[label], list) or isinstance(y_pred[label], list):
            data += similarity_zip(y_true[label], y_pred[label])
        else:
            data += [(y_true[label], y_pred[label])]
    similar_observations = sum(
        similarity_score(y_true, y_pred) > threshold
        for y_true, y_pred in data
    )
    print(similar_observations, len(data))
    return similar_observations / len(data)


def precision_recall_f1_score(metrics, type='exact'):
    pos = metrics.loc[['COR', 'INC', 'PAR', 'MIS']].sum()
    act = metrics.loc[['COR', 'INC', 'PAR', 'SPU']].sum()

    num = (metrics.loc['COR']
           if type == 'exact'
           else metrics.loc['COR'] + 0.5 * metrics.loc['PAR'])
    
    precision, recall = num.div(act), num.div(pos)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return pd.DataFrame(
        [precision, recall, f1_score],
        columns=metrics.columns,
        index=['Precision', 'Recall', 'F1 Score']
    )


def get_metric(span1, span2, type1=None, type2=None, type='exact'):
    _type1 = type1.lower() if isinstance(type1, str) else type1 or None
    _type2 = type2.lower() if isinstance(type2, str) else type2 or None
    _span1 = span1.lower().strip() if isinstance(span1, str) else span1 or None
    _span2 = span2.lower().strip() if isinstance(span2, str) else span2 or None
    if type != 'type':
        if _span1 == _span2:
            metric = (
                'COR'
                if type in ['exact', 'partial'] \
                    or (type == 'strict' and _type1 == _type2)
                else 'INC')
        elif _span1 is None:
            metric = 'SPU'
        elif _span2 is None:
            metric = 'MIS'
        else:
            metric = 'INC'
            if type == 'partial':
                similarity = similarity_meassure(_span1, _span2)
                metric = 'PAR' if similarity > 0.5 else 'INC'
    else:
        similarity = similarity_score(_span1, _span2)
        if similarity <= 0.5:
            metric = 'INC'
        elif _type1 == _type2:
            metric = 'COR'
        elif _type1 is None:
            metric = 'SPU'
        elif _type2 is None:
            metric = 'MIS'
        else:
            metric = 'INC'
    return metric


def similarity_zip(iter1, iter2):
    zipped_pairs, paired, available = [], [], list(range(len(iter2 or [])))
    for i in range(len(iter1)):
        max_score, max_idx = 0, 0
        for j in available:
            score = similarity_score(iter1[i], iter2[j])
            if score == 1:
                max_score, max_idx = 1, j
                break
            if score > max_score:
                max_score, max_idx = score, j
        if max_score >= 0.5:
            zipped_pairs.append((iter1[i], iter2[max_idx]))
            paired.append(i)
            available.remove(max_idx)
    not_paired = list(set(range(len(iter1))).difference(paired))
    if not_paired:
        zipped_pairs += [(iter1[i], iter2[j]) for i, j in zip(not_paired, available)]
    if len(iter1) == len(iter2 or []) == 0 and not iter1:
        zipped_pairs += [(None, None)]
    if len(iter1) > len(iter2 or []):
        zipped_pairs += [(iter1[i], None) for i in not_paired[len(available):]]
    if len(iter1) < len(iter2 or []):
        zipped_pairs += [(None, iter2[i]) for i in available[len(not_paired):]]
    return zipped_pairs


def match_score(true_data, predict_data, type='exact', labels=None):
    labels = labels or ['subject', 'object', 'event', 'complement']

    metrics_table = pd.DataFrame(
        [[0] * (len(labels) + 1)] * 5,
        columns=[label.title() for label in labels] + ['Total'],
        index=['COR', 'INC', 'PAR', 'MIS', 'SPU']
    )

    for y_true, y_pred in zip(true_data, predict_data):
        if y_true['text'].strip() != y_pred['text'].strip():
            print('!!!')
        if type in ('strict', 'type'):
            types = {
                'subject': (
                    'subjectLabel' in y_true and y_true['subjectLabel'], y_pred['subjectLabel']),
                'object': (
                    'objectLabel' in y_true and y_true['objectLabel'], y_pred['objectLabel']),
                'event': (
                    'relationType' in y_true and y_true['relationType'], y_pred['relationType'])
            }

        for label in labels:
            label_pairs = [(y_true[label], y_pred[label])]
            type_true, type_pred = None, None
            if isinstance(label_pairs[0][0], list):
                label_pairs = list(similarity_zip(*label_pairs[0]))
            if label != 'complement' and type in ('strict', 'type'):
                type_true, type_pred = types[label]
                if type_pred == 'None': type_pred = None
            for label_true, label_pred in label_pairs:
                metric = get_metric(
                    label_true, label_pred, type_true, type_pred, type=type)
                # if metric in ('INC') and label == 'subject':
                #     print(label_pairs, type_true, type_pred)
                #     print(y_true['text'].strip())
                metrics_table.loc[metric, label.title()] += 1

    metrics_table['Total'] = metrics_table[[
        label.title() for label in labels]].sum(1)

    metric_type = 'exact' if type in ['exact', 'strict'] else ['partial', 'type']

    metrics_table = pd.concat([
        metrics_table,
        precision_recall_f1_score(metrics_table, type=metric_type)
    ])

    return metrics_table.fillna(0)


def match_relation_type_score(true_data, predict_data):
    tp, tn, fp, fn = 0, 0, 0, 0
    score_per_type = {}

    for y_true, y_pred in zip(true_data, predict_data):
        type1, type2 = y_true['relationType'], y_pred['relationType']
        type1 = type1.lower() if isinstance(type1, str) else type1
        type2 = type2.lower() if isinstance(type2, str) else type2
        if (type1 is None or type1 == 'norelation') and (type2 is None or type2 == 'norelation'):
            tn += 1
        if type2 is None or type2 == 'norelation':
            fn += 1
        if type1 == type2:
            tp += 1
        fp += 1
    print(tp, tn, fp, fn)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * (p * r) / (p + r)
    return (p, r, f1)


def confusion_matrix(true_data, predict_data, label, none_class='none'):
    classes = list(set(sentence[label].lower() for sentence in true_data if label in sentence))
    if none_class not in classes:
        classes.append(none_class)
    cm = pd.DataFrame(
        [[0] * len(classes)] * len(classes),
        columns=classes,
        index=classes,
    )

    for y_true, y_pred in zip(true_data, predict_data):
        type1, type2 = y_true[label] if label in y_true else none_class, y_pred[label]
        type1 = type1.lower() if isinstance(type1, str) else str(type1).lower()
        type2 = type2.lower() if isinstance(type2, str) else str(type2).lower()

        cm.loc[type1, type2] += 1

    return cm
