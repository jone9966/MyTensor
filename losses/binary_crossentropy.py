import numpy as np


def binary_crossentropy(y, yhat):
    return np.sum(- (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))) / y.shape[1]
