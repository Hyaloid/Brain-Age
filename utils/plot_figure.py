import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from typing import Union
from sklearn.linear_model import Lasso


# 查看数据分布和正态分布的差别
def dist_plt(data, idx: Union[str, int]):
    sns.distplot(data[idx], fit=norm)

    (mu, sigma) = norm.fit(data[idx])
    print(f'\n mu = {mu:.2f} and sigma = {sigma:.2f}\n')
    plt.legend([f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
    plt.ylabel(idx)
    plt.title('distribution')

    plt.show()


def prob_plt(data, idx: Union[str, int]):
    plt.figure()
    stats.probplot(data[idx], plot=plt)
    plt.show()


# 检查数据相关性
def check_correlation(train):
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()


def feature_selection_lasso(train, y_train):
    lasso = Lasso(alpha=10 ** (-3))
    model_lasso = lasso.fit(train, y_train)
    coef = pd.Series(model_lasso.coef_, index=train.columns)
    print(coef[coef != 0].abs().sort_values(ascending=False))

    # 打印特征选择结果
    selected_features = np.where(lasso.coef_ != 0)[1]
    print("Selected features:", selected_features)

    fea = train.columns
    a = pd.DataFrame()
    a['feature'] = fea
    a['importance'] = coef.values

    # res_df = pd.DataFrame(res)
    a[a['importance'] > 0].to_csv('feature_greater_than_zero.csv', index=False)

    # plt bar graph
    a = a.sort_values('importance', ascending=False)
    plt.figure(figsize=(30, 60))
    plt.barh(a['feature'], a['importance'])
    plt.title('the importance features')
    plt.show()

