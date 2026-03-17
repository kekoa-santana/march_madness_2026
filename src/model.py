import numpy as np


def brier_score(y_true, y_pred):
    '''Mean squared error between true labels and predicted probabilities.'''
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
