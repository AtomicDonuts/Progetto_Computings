'''
doc    
'''
import numpy as np
import sklearn.metrics as sk_metrics


def best_accuracy(true_labels, predicted_labels):
    """Trova la soglia che massimizza l'accuratezza sul training set."""
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_acc = 0.0
    for t in thresholds:
        th_pred = (predicted_labels >= t).astype(int)
        acc = sk_metrics.accuracy_score(true_labels, th_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t
    return best_acc, best_threshold


def best_eq_accuracy(true_labels, predicted_labels):
    """Trova la soglia che minimizza la differenza di accuratezza tra le classi."""
    thresholds = np.linspace(0, 1, 10001)
    best_threshold = 0.5
    min_diff = 1.0
    best_agn = 0
    best_psr = 0
    for t in thresholds:
        th_pred = (predicted_labels >= t).astype(int)
        cm = sk_metrics.confusion_matrix(true_labels, th_pred)
        tn, fp, fn, tp = cm.ravel()
        acc_agn = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc_psr = tp / (tp + fn) if (tp + fn) > 0 else 0
        diff = abs(acc_agn - acc_psr)
        if diff < min_diff:
            min_diff = diff
            best_agn = acc_agn
            best_psr = acc_psr
            best_threshold = t
    return best_agn, best_psr, best_threshold


def best_f1_score(true_labels, predicted_labels):
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in thresholds:
        th_pred = (predicted_labels >= t).astype(int)
        cm = sk_metrics.confusion_matrix(true_labels, th_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        f1 = tp / (tp + 0.5 * (fp + fn))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_f1, best_threshold


def accuracy(th, true_labels, predicted_labels):
    th_pred = (predicted_labels >= th).astype(int)
    acc = sk_metrics.accuracy_score(true_labels, th_pred)
    return acc


def class_accuracy(th, true_labels, predicted_labels):
    th_pred = (predicted_labels >= th).astype(int)
    cm = sk_metrics.confusion_matrix(true_labels, th_pred)
    tn, fp, fn, tp = cm.ravel()
    acc_agn = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_psr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return acc_agn, acc_psr


def f1_score(th, true_labels, predicted_labels):
    th_pred = (predicted_labels >= th).astype(int)
    cm = sk_metrics.confusion_matrix(true_labels, th_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1
