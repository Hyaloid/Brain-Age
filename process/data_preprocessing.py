import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

cur_dir = os.getcwd()
print(cur_dir)


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore warning

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 保留浮点数三位


# 导入数据
train = pd.read_csv('../data/input/data_train.csv')
test = pd.read_csv('../data/input/data_test.csv')


# 查看异常值
fig, ax = plt.subplots()
ax.scatter(x=train['Right-UnsegmentedWhiteMatter'], y=train['年龄'])
plt.ylabel('', fontsize=13)
plt.xlabel('Right-UnsegmentedWhiteMatter', fontsize=13)

plt.show()


# if __name__ == '__main__':
#
#     # 显示训练集前五行
#     print(train.head(5))
#     print(test.head(5))
#
#     print(train.shape)  # 1600 x 411
#     print(test.shape)   # 389 x 410
