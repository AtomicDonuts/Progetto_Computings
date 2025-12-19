"""
This module contains custom metric functions for evaluating model performance.
It includes utilities to calculate accuracy, F1 score, and find optimal thresholds
for classification.
"""

import numpy as np
import sklearn.metrics as sk_metrics


def best_accuracy(true_labels, predicted_labels):
    """
    Finds the probability threshold that maximizes accuracy on the given set.

    :param true_labels: The ground truth binary labels (0 or 1).
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities from the model.
    :type predicted_labels: numpy.ndarray
    :return: A tuple containing the best accuracy achieved and the corresponding threshold.
    :rtype: tuple(float, float)
    """
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
    """
    Finds the threshold that minimizes the difference between the accuracy of the two classes.
    This is useful for balancing performance on unbalanced datasets.

    :param true_labels: The ground truth binary labels.
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities.
    :type predicted_labels: numpy.ndarray
    :return: A tuple containing (Accuracy Class 0, Accuracy Class 1, Best Threshold).
    :rtype: tuple(float, float, float)
    """
    thresholds = np.linspace(0, 1, 10001)
    best_threshold = 0.5
    min_diff = 1.0
    best_tnr = 0
    best_tpr = 0
    for t in thresholds:
        th_pred = (predicted_labels >= t).astype(int)
        cm = sk_metrics.confusion_matrix(true_labels, th_pred)
        tn, fp, fn, tp = cm.ravel()
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        diff = abs(true_negative_rate - true_positive_rate)
        if diff < min_diff:
            min_diff = diff
            best_tnr = true_negative_rate
            best_tpr = true_positive_rate
            best_threshold = t
    return best_tnr, best_tpr, best_threshold


def best_f1_score(true_labels, predicted_labels):
    """
    Finds the threshold that maximizes the F1 Score.

    :param true_labels: The ground truth binary labels.
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities.
    :type predicted_labels: numpy.ndarray
    :return: A tuple containing the best F1 score and the corresponding threshold.
    :rtype: tuple(float, float)
    """
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
    """
    Calculates the accuracy score for a specific threshold.

    :param th: The decision threshold to apply.
    :type th: float
    :param true_labels: The ground truth binary labels.
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities.
    :type predicted_labels: numpy.ndarray
    :return: The accuracy score.
    :rtype: float
    """
    th_pred = (predicted_labels >= th).astype(int)
    acc = sk_metrics.accuracy_score(true_labels, th_pred)
    return acc


def class_accuracy(th, true_labels, predicted_labels):
    """
    Calculates the accuracy individually for both classes.

    :param th: The decision threshold to apply.
    :type th: float
    :param true_labels: The ground truth binary labels.
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities.
    :type predicted_labels: numpy.ndarray
    :return: A tuple containing (Accuracy Class 0, Accuracy Class 1).
    :rtype: tuple(float, float)
    """
    th_pred = (predicted_labels >= th).astype(int)
    cm = sk_metrics.confusion_matrix(true_labels, th_pred)
    tn, fp, fn, tp = cm.ravel()
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    return true_negative_rate, true_positive_rate


def f1_score(th, true_labels, predicted_labels):
    """
    Calculates the F1 score for a specific threshold.

    :param th: The decision threshold to apply.
    :type th: float
    :param true_labels: The ground truth binary labels.
    :type true_labels: numpy.ndarray
    :param predicted_labels: The continuous prediction probabilities.
    :type predicted_labels: numpy.ndarray
    :return: The F1 score.
    :rtype: float
    """
    th_pred = (predicted_labels >= th).astype(int)
    cm = sk_metrics.confusion_matrix(true_labels, th_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1 = tp / (tp + 0.5 * (fp + fn))
    return f1

def roc_curve(true_labels,predicted_labels):
    thresholds = np.linspace(0, 1, 101)
    roc_curve_array = []
    for th in thresholds:
        th_pred = (predicted_labels >= th).astype(int)
        cm = sk_metrics.confusion_matrix(true_labels, th_pred)
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_curve_array.append((false_positive_rate,true_positive_rate))
    return roc_curve_array
