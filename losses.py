import numpy as np
eps = 1e-15
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    tmp = np.mean(-y_true * np.log(np.maximum(y_pred , eps)) - (1 - y_true) * np.log(np.maximum(1 - y_pred , eps)))
    # print("binary_cross_entropy", tmp)
    return tmp

def binary_cross_entropy_prime(y_true, y_pred):
    # print("y_true", y_true)
    # print("y_pred", y_pred)
    tmp = ((1 - y_true) / (np.maximum(1 - y_pred , eps)) - y_true / np.maximum(y_pred , eps)) / (y_true.shape[0] * y_true.shape[1])
    # print("binary_cross_entropy_prime", tmp)
    return tmp
