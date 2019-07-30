import numpy as np
"""
涉及参数优化的相关函数
优化方法我们使用随机梯度下降（SGD）
"""

def _copy_weights_to_zeros(weights):
    result = {}
    result.keys()
    for key in weights.keys():
        result[key] = np.zeros_like(weights[key])
    return result

##随机梯度下降
class SGD(object):
    """
    小批量梯度下降法
    """

    def __init__(self, weights, lr=0.01, momentum=0.9, decay=1e-5):
        """
        :param weights: 权重，字典类型
        :param lr: 初始学习率
        :param momentum: 动量因子
        :param decay: 学习率衰减
        """
        self.v = _copy_weights_to_zeros(weights)  # 累积动量大小
        self.iterations = 0  # 迭代次数
        self.lr = self.init_lr = lr
        self.momentum = momentum
        self.decay = decay

    def iterate(self, weights, gradients):
        """
        迭代一次
        :param weights: 当前迭代权重
        :param gradients: 当前迭代梯度
        :return:
        """
        # 更新学习率
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        # 更新动量和梯度
        for key in self.v.keys():
            self.v[key] = self.momentum * self.v[key] + self.lr * gradients[key]
            weights[key] = weights[key] - self.v[key]

        # 更新迭代次数
        self.iterations += 1

