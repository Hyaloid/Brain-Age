# coding=utf8
import pandas as pd
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 保留浮点数三位

# 导入数据
train = pd.read_csv('../dataset/input/data_train.csv')
test = pd.read_csv('../dataset/input/data_test.csv')


# 查看异常值
for col in train:
    if col == "subject_ID":
        continue
    fig, ax = plt.subplots()
    ax.scatter(x=train[col], y=train['年龄'])
    plt.ylabel('年龄', fontsize=13)
    plt.xlabel(col, fontsize=13)
    plt.show()


# 查看数据分布和正态分布的差别
sns.distplot(train['Age'], fit=norm)

(mu, sigma) = norm.fit(train['lh_GausCurv_caudalanteriorcingulate'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('lh_GausCurv_caudalanteriorcingulate')
plt.title('Age distribution')

fig = plt.figure()
res = stats.probplot(train['lh_GausCurv_caudalanteriorcingulate'], plot=plt)
plt.show()


# 检查数据相关性
corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
