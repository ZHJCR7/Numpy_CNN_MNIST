import gzip
from six.moves import cPickle as pickle
import os
import platform
"""
加载mnist数据集，
数据集的具体描述，
可以参考博客：https://www.cnblogs.com/upright/p/4191757.html
"""

#选Python3加载cPickle
def load_pickle(f):
    pickle.load(f, encoding='latin1')

#加载mnist数据，返回训练，测试，验证集
def load_mnist_datasets(path='./Data/mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = load_pickle(f)#训练，验证和测试
        return train_set, val_set, test_set
