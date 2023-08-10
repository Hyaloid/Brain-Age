import numpy as np
import pandas as pd
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import warnings
import os

from scipy import stats
from scipy.stats import norm, skew

cur_dir = os.getcwd()
print(cur_dir)


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore warning

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 保留浮点数三位

# 导入数据
train = pd.read_csv('../data/input/data_train.csv')
test = pd.read_csv('../data/input/data_test.csv')
# train = pd.read_csv('../data/tmp_input/train.csv')
# test = pd.read_csv('../data/tmp_input/test.csv')

print([col for col in train])

# 查看异常值
# for col in train:
#     if col == "subject_ID":
#         continue
#     fig, ax = plt.subplots()
#     ax.scatter(x=train[col], y=train['年龄'])
#     plt.ylabel('年龄', fontsize=13)
#     plt.xlabel(col, fontsize=13)
#     plt.show()

# print(train[(train['wm-rh-frontalpole'] > 700) & (train['年龄'] < 80)].index)

sns.distplot(train['年龄'], fit=norm)

(mu, sigma) = norm.fit(train['lh_GausCurv_caudalanteriorcingulate'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('lh_GausCurv_caudalanteriorcingulate')
plt.title('Age distribution')

fig = plt.figure()
res = stats.probplot(train['lh_GausCurv_caudalanteriorcingulate'], plot=plt)
plt.show()

# if __name__ == '__main__':
#
#     # 显示训练集前五行
#     print(train.head(5))
#     print(test.head(5))
#
#     print(train.shape)  # 1600 x 411
#     print(test.shape)   # 389 x 410
