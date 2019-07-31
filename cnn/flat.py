import numpy as np

#将多维数组展平，前向传播
def flatten_forward(X):
    """
    :param X: 多维数组,形状(N,d1,d2,..)
    :return:
    """
    N = X.shape[0]
    return np.reshape(X, (N, -1))

#打平层反向传播
def flatten_backward(next_dX, X):
    """
    :param next_dX:
    :param X:
    :return:
    """
    return np.reshape(next_dX, X.shape)