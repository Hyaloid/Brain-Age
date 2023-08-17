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
bad_feature = ['lh_MeanCurv_transversetemporal', 'lh_MeanCurv_insula', 'lh_GausCurv_medialorbitofrontal', 'lh_GausCurv_superiorfrontal', 'lh_GausCurv_superiortemporal', 'lh_GausCurv_supramarginal', 'lh_GausCurv_transversetemporal', 'wm-lh-rostralanteriorcingulate', 'wm-lh-frontalpole', 'wm-lh-transversetemporal', 'wm-rh-entorhinal', 'wm-rh-isthmuscingulate', 'wm-rh-parahippocampal', 'wm-rh-posteriorcingulate', 'wm-rh-rostralanteriorcingulate', 'wm-rh-temporalpole', 'wm-rh-transversetemporal', 'wm-rh-insula', 'Left-UnsegmentedWhiteMatter', 'Right-UnsegmentedWhiteMatter']
fea = []
# for f in train:
#     print(f)
    # fea.append(f)

not_gauss = []
train.drop(['subject_ID'], axis=1, inplace=True)

from scipy import stats
import numpy as np
for col in train:
    # print('Kolmogorov-Smirnov Test Statistic: ', statistic)
    # print('P-value: ', p_value)
    statistic, p_value = stats.shapiro(train[col])
    alpha = 0.05
    if p_value > alpha:
        print(f'{col} sample looks gauss')
    else:
        try:
            data = np.log1p(train[col])
            data2 = np.log1p(test[col])
            statistic, p_value = stats.shapiro(data)
            if p_value > alpha:
                train[col] = data
                test[col] = data2
                print(f'{col} sample looks gauss after log')
            else:
                not_gauss.append(col)
        except:
            ...

print(len(train))
print(len(not_gauss))
