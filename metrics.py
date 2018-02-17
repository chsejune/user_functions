__author__ = 'Sejune Cheon'
__version__ = '1.0'
__environment__ = 'Python-3.5.2, numpy-1.13.3'
__description__ = 'metrics function collection'


import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
