import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm


# 查看数据分布和正态分布的差别
def dist_plt(train):
    sns.distplot(train['Age'], fit=norm)

    (mu, sigma) = norm.fit(train['lh_GausCurv_caudalanteriorcingulate'])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('lh_GausCurv_caudalanteriorcingulate')
    plt.title('Age distribution')

    plt.show()


# 检查数据相关性
def check_correlation(train):
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
