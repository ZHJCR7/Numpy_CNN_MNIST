import numpy as np

"""
定义关于激活函数Relu的前向反向传播
"""

def relu_forward(X):
    """
    relu前向传播
    :param X: 待激活层
    :return: 激活后的结果
    """
    return np.maximum(0, X)


def relu_backward(next_dX, X):
    """
    relu反向传播
    :param next_dX: 激活后的梯度
    :param X: 激活前的值
    :return:
    """
    dX = np.where(np.greater(X, 0), next_dX, 0)
    return dX

