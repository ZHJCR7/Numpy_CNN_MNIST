import numpy as np

#将多维数组展平，前向传播
def flatten_forward(z):
    """
    :param z: 多维数组,形状(N,d1,d2,..)
    :return:
    """
    N = z.shape[0]
    return np.reshape(z, (N, -1))

#打平层反向传播
def flatten_backward(next_dz, z):
    """
    :param next_dz:
    :param z:
    :return:
    """
    return np.reshape(next_dz, z.shape)