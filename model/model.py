# coding=utf8
from model_class import StackingAverageModels
from utils.score import model_train, model_pred
from utils.save_file import write2csv
from dataset.load_data import train, test
import pandas as pd

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
train = pd.concat([train.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri, encodes_sex], axis=1)

# Test MRI & Sex 哑变量处理
encoded_mri = pd.get_dummies(test['MRI扫描仪类型'], prefix='MRI')
encodes_sex = pd.get_dummies(test['性别'], prefix='Sex')
test = pd.concat([test.drop(['MRI扫描仪类型', '性别'], axis=1), encoded_mri, encodes_sex], axis=1)


# stacking model
stacked_averaged_model1 = StackingAverageModels(base_models=(lasso, model_lgb), meta_model=model_lgb)
stacked_averaged_model2 = StackingAverageModels(base_models=(stacked_averaged_model1, model_lgb), meta_model=lasso)

model_train(stacked_averaged_model2, train, y_train, model_name='stacked_averaged_model1')

stacked_averaged_model2.fit(train.values, y_train)
pred = stacked_averaged_model2.predict(test.values)
# pred = model_pred(test, 'stacked_averaged_model1.pkl')
write2csv(test_id, pred, filename='submission.csv')

