from constants import NP_DATASET_PATH
import numpy as np


def shuffle_data(X, y):
    p = np.random.permutation(len(X))
    return X[p], y[p]


def get_dataset(path=NP_DATASET_PATH, dtype_data=None, dtype_label=None, shuffle=True):
    data = np.load(path)
    X, y = data['X'], data['y']
    if dtype_data:
        X = X.astype(dtype_data)
    if dtype_label:
        y = y.astype(dtype_label)
    if shuffle:
        return shuffle_data(X, y)
    else:
        return X, y