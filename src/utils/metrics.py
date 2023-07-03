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
    return np.average([
        similarity_score(y_true[label], y_pred[label])
        for y_true, y_pred in zip(true_data, predict_data)
    ])


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
    if type != 'type':
        if span1 == span2:
            metric = (
                'COR'
                if type in ['exact', 'partial'] \
                    or (type == 'strict' and type1 == type2)
                else 'INC')
        elif span1 is None:
            metric = 'SPU'
        elif span2 is None:
            metric = 'MIS'
        else:
            metric = 'INC'
            if type == 'partial':
                similarity = similarity_meassure(span1, span2)
                metric = 'PAR' if similarity > 0.5 else 'INC'
    else:
        similarity = similarity_score(span1, span2)
        if similarity <= 0.5:
            metric = 'INC'
        elif type1 == type2:
            metric = 'COR'
        elif type1 is None:
            metric = 'SPU'
        elif type2 is None:
            metric = 'MIS'
        else:
            metric = 'INC'
    return metric


def match_score(true_data, predict_data, type='exact'):
    metrics_table = pd.DataFrame(
        [[0] * 4] * 5,
        columns=['Subject', 'Object', 'Event', 'Total'],
        index=['COR', 'INC', 'PAR', 'MIS', 'SPU']
    )

    for y_true, y_pred in zip(true_data, predict_data):
        types = {
            'subject': (y_true['subjectLabel'], y_pred['subjectLabel']),
            'object': (y_true['objectLabel'], y_pred['objectLabel']),
            'event': (y_true['relationType'], y_pred['relationType'])
        }

        for label in ['subject', 'object', 'event']:
            label_true, label_pred = y_true[label], y_pred[label]
            type_true, type_pred = types[label]
            metric = get_metric(
                label_true, label_pred, type_true, type_pred, type=type)
            metrics_table.loc[metric, label.title()] += 1

    metrics_table['Total'] = metrics_table[[
        'Subject', 'Object', 'Event']].sum(1)
    
    metric_type = 'exact' if type in ['exact', 'strict'] else ['partial', 'type']

    metrics_table = pd.concat([
        metrics_table,
        precision_recall_f1_score(metrics_table, type=metric_type)
    ])

    return metrics_table
