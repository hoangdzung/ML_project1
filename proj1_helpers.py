# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def acc_score(actual, predicted):
    """
    Accuracy classification score.
    
    Parameters
    ----------
    actual : 1d numpy.ndarray
        Ground truth (correct) labels.
    predicted : 1d numpy.ndarray
        Predicted labels
    Returns
    -------
    float:
        accuracy score
    """

    return (actual==predicted).mean()

def f1_score(actual, predicted, label=1):

    """
    F1 score.
    
    Parameters
    ----------
    actual : 1d numpy.ndarray
        Ground truth (correct) labels.
    predicted : 1d numpy.ndarray
        Predicted labels
    label: int
        Report scores for that label only, default 1
        Usually in binary classification, the positive label is more critical
    Returns
    -------
    float:
        f1 score
    """
    
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))
    
    # add very small numbers to avoid division by zero
    precision = tp/(tp+fp+1e-20)
    recall = tp/(tp+fn+1e-20)
    f1 = 2 * (precision * recall) / (precision + recall+1e-20)
    return f1