# coding=utf8
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import numpy as np


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))  # 保留浮点数三位
color = sns.color_palette()
sns.set_style('darkgrid')

# 导入数据

bad_feature = ['lh_MeanCurv_transversetemporal', 'lh_MeanCurv_insula', 'lh_GausCurv_medialorbitofrontal',
               'lh_GausCurv_superiorfrontal', 'lh_GausCurv_superiortemporal', 'lh_GausCurv_supramarginal',
               'lh_GausCurv_transversetemporal', 'wm-lh-rostralanteriorcingulate', 'wm-lh-frontalpole',
               'wm-lh-transversetemporal', 'wm-rh-entorhinal', 'wm-rh-isthmuscingulate', 'wm-rh-parahippocampal',
               'wm-rh-posteriorcingulate', 'wm-rh-rostralanteriorcingulate', 'wm-rh-temporalpole',
               'wm-rh-transversetemporal', 'wm-rh-insula', 'Left-UnsegmentedWhiteMatter',
               'Right-UnsegmentedWhiteMatter']


def check_gauss_dist(train_, test_):
    not_gauss = []
    for col in train_:
        # print('Kolmogorov-Smirnov Test Statistic: ', statistic)
        # print('P-value: ', p_value)
        statistic, p_value = stats.shapiro(train_[col])
        alpha = 0.05
        if p_value > alpha:
            print(f'{col} sample looks gauss')
        else:
            try:
                data_train = np.log1p(train_[col])
                data_test = np.log1p(test_[col])
                statistic, p_value = stats.shapiro(data_train)
                if p_value > alpha:
                    train_[col] = data_train
                    test_[col] = data_test
                    print(f'{col} sample looks gauss after log')
                else:
                    not_gauss.append(col)
            except ValueError:
                ...


def data_preprocessing(train_, test_):
    # preprocessing
    y_train = train_.Age.values

    train_.drop(['subject_ID', 'Age'], axis=1, inplace=True)

    test_id = test_['subject_ID']
    test_.drop(['subject_ID'], axis=1, inplace=True)

    # Train MRI & Sex 哑变量处理
    encoded_mri = pd.get_dummies(train_['MRI扫描仪类型'], prefix='MRI')
    # encoded_sex = pd.get_dummies(train_['性别'], prefix='Sex', dtype=int)
    train_ = pd.concat([train_.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri], axis=1)

    # Test MRI & Sex 哑变量处理
    encoded_mri = pd.get_dummies(test_['MRI扫描仪类型'], prefix='MRI')
    # encoded_sex = pd.get_dummies(test_['性别'], prefix='Sex', dtype=int)
    test_ = pd.concat([test_.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri], axis=1)
    return train_, y_train, test_, test_id


def standardize_data(train_, test_):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_)
    test_scaled = scaler.transform(test_)
    return train_scaled, test_scaled
