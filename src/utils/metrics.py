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


def calculate_metrics(confusion_matrix, as_df=False):
    tp, tn, fp, fn = confusion_matrix['TP'], confusion_matrix['TN'], confusion_matrix['FP'], confusion_matrix['FN']
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * (precision * recall) / (precision + recall)
          if precision + recall
          else 0)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return pd.DataFrame(metrics, index=['metrics']).T if as_df else metrics


def evaluate_label_metrics(true_data, predict_data, label, threshold=0.5):
    confusion_matrix = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    for y_true, y_pred in zip(true_data, predict_data):
        if y_true[label] == y_pred[label]:
            metric = 'TN' if y_true is None else 'TP'
        elif y_true[label] is None or y_pred[label] is None:
            metric = 'FP' if y_true is None else 'FN'
        else:
            metric = (
                'FP'
                if similarity_meassure(y_true[label], y_pred[label]) <= threshold
                else 'TP')

        confusion_matrix[metric] += 1
    return confusion_matrix
