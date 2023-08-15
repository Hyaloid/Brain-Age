# coding=utf8
from model_class import AveragingModels
from utils.score import rmsle, rmsle_cv, mae_cv, calculate_mae, model_train, model_pred
from utils.save_file import write2csv
from dataset.load_data import train, test
import pandas as pd
import numpy as np

from base_models import *

n_train = train.shape[0]
n_test = test.shape[0]
tmp_test = test

# preprocessing
y_train = train.Age.values

train.drop(['subject_ID', 'Age'], axis=1, inplace=True)

test_id = test['subject_ID']
test.drop(['subject_ID'], axis=1, inplace=True)

# Train MRI & Sex 哑变量处理
encoded_mri = pd.get_dummies(train['MRI扫描仪类型'], prefix='MRI')
encodes_sex = pd.get_dummies(train['性别'], prefix='Sex')
# train = pd.concat([train.drop(['MRI扫描仪类型'], axis=1), encoded_mri], axis=1)
train = pd.concat([train.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri, encodes_sex], axis=1)

# Test MRI & Sex 哑变量处理
encoded_mri = pd.get_dummies(test['MRI扫描仪类型'], prefix='MRI')
encodes_sex = pd.get_dummies(test['性别'], prefix='Sex')
# test = pd.concat([test.drop(['MRI扫描仪类型'], axis=1), encoded_mri], axis=1)
test = pd.concat([test.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri, encodes_sex], axis=1)

# print(train.columns)
# print(test.columns)

averaged_models = AveragingModels(models=(GBoost, KRR, model_lgb, lasso))
model_train(averaged_models, train, y_train, model_name='GBoost_KRR_LGB_lasso')
pred = model_pred(test, 'GBoost_KRR_LGB_lasso.pkl')
pred[pred < 0] = 0


write2csv(test_id, pred)
# model_pred(test, file_name='GBoost_KRR_LGB_lasso.pkl')
