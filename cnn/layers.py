import numpy as np

"""
layers中定义前向传播和反向传播
"""

#全连接层的前向传播
def fullyconnected_forward(z, W, b):
    """
    :param z: 当前层的输出,形状 (N,ln)
    :param W: 当前层的权重
    :param b: 当前层的偏置
    :return: 下一层的输出
    """
    return np.dot(z, W) + b

#全连接层的反向传播
def fullyconnected_backward(next_dz, W, z):
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
def _remove_padding(X, padding):
    """
    :param X: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return X[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return X[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return X[:, :, :, padding[1]:-padding[1]]
    else:
        return X

#想多维数组最后两位，每个行列之间增加指定的个数的零填充
def _insert_zeros(dX, strides):
    """
    :param dX: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dX.shape
    pX = dX
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pX = np.insert(pX, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pX = np.insert(pX, w, 0, axis=3)
    return pX

#关于卷积层和池化层的前向和反向传播
##关于卷积层前向和反向传播的实现测试见Test_hahaha.ipynb
def convolution_forward(X_input, Kernel, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param Kernel: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_X = np.lib.pad(X_input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_X.shape
    C, D, k1, k2 = Kernel.shape

    ##简单设计，防止出现不能整除情况，可用floor函数避免
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'

    ##卷积之后的长度，padding为0
    H_ = 1 + (height - k1) // strides[0]
    W_ = 1 + (width - k2) // strides[1]
    conv_X = np.zeros((N, D, H_, W_))

    ##求和操作
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_X[n, d, h // strides[0], w // strides[1]] = np.sum(
                        padding_X[n, :, h:h + k1, w:w + k2] * Kernel[:, d]) + b[d]
    return conv_X

def convolution_backward(next_dX, Kernel, X, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    :param next_dX: 卷积输出层的梯度,(N,D,H',W'),H',W'为卷积输出层的高度和宽度
    :param Kernel: 当前层卷积核，(C,D,k1,k2)
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param padding: padding
    :param strides: 步长
    :return:
    """
    N, C, H, W = X.shape
    C, D, k1, k2 = Kernel.shape

    # 卷积核梯度
    padding_next_dX = _insert_zeros(next_dX, strides)

    # 卷积核高度和宽度翻转180度
    flip_K = np.flip(Kernel, (2, 3))
    # 交换C,D为D,C；D变为输入通道数了，C变为输出通道数了
    swap_flip_K = np.swapaxes(flip_K, 0, 1)
    # 增加高度和宽度0填充
    ppadding_next_dX = np.lib.pad(padding_next_dX, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant', constant_values=0)
    ##rot(180)*W
    dX = convolution_forward(ppadding_next_dX.astype(np.float64), swap_flip_K.astype(np.float64), np.zeros((C,), dtype=np.float64))

    # 求卷积核的梯度dK
    swap_W = np.swapaxes(X, 0, 1)  # 变为(C,N,H,W)与
    dW = convolution_forward(swap_W.astype(np.float64), padding_next_dX.astype(np.float64), np.zeros((D,), dtype=np.float64))

    # 偏置的梯度
    db = np.sum(np.sum(np.sum(next_dX, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

    # 把padding减掉
    dX = _remove_padding(dX, padding)

    return dW / N, db / N, dX

def maxpooling_forward(X, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = X.shape
    # 零填充
    padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)

    # 输出的高度和宽度
    H_ = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    W_ = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_X = np.zeros((N, C, H_, W_))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(H_):
                for j in np.arange(W_):
                    ##参考公式中i*s< <i*s+k
                    pool_X[n, c, i, j] = np.max(padding_X[n, c,
                                                          strides[0] * i:strides[0] * i + pooling[0],
                                                          strides[1] * j:strides[1] * j + pooling[1]])
    return pool_X

def maxpooling_backward(next_dX, X, pooling, strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    :param next_dX：损失函数关于最大池化输出的损失
    :param X: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = X.shape
    _, _, H_, W_ = next_dX.shape
    # 零填充
    padding_X = np.lib.pad(X, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 零填充后的梯度
    padding_dX = np.zeros_like(padding_X)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(H_):
                for j in np.arange(W_):
                    # 找到最大值的那个元素坐标，将梯度传给这个坐标
                    # 参考公式s1*i+k1和s2*j+k2
                    flat_idx = np.argmax(padding_X[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dX[n, c, h_idx, w_idx] += next_dX[n, c, i, j]
    # 返回时剔除零填充
    return _remove_padding(padding_dX, padding)


