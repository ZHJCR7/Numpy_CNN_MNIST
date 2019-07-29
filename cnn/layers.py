import numpy as np

"""
layers中定义前向传播和反向传播
"""

#全连接层的前向传播
def fc_forward(z, W, b):
    """
    :param z: 当前层的输出,形状 (N,ln)
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(z, W) + b

#全连接层的反向传播
def fc_backward(next_dz, W, z):
    """
    :param next_dz: 下一层的梯度
    :param W: 当前层的权重
    :param z: 当前层的输出
    :return:
    """
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)  # 当前层的梯度
    dw = np.dot(z.T, next_dz)  # 当前层权重的梯度
    db = np.sum(next_dz, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
    return dw / N, db / N, dz

#移除padding
def _remove_padding(z, padding):
    """
    :param z: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z

#想多维数组最后两位，每个行列之间增加指定的个数的零填充
def _insert_zeros(dz, strides):
    """
    :param dz: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz

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



