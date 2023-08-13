from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from modeling import AveragingModels
import xgboost as xgb
import lightgbm as lgb
import pandas as pd

import numpy as np

# from preprocessing.data_preprocessing import train, test

train = pd.read_csv('../dataset/input/data_train.csv')
test = pd.read_csv('../dataset/input/data_test.csv')

n_train = train.shape[0]
n_test = test.shape[0]

# Validation function
n_folds = 5

train.drop(['subject_ID'], axis=1, inplace=True)

train_na = (train.isnull().sum() / len(train)) * 100
missing_data = pd.DataFrame({'Missing Ration': train_na})
train['性别'] = train['性别'].fillna(2.0)
y_train = train.Age.values

# import matplotlib.pyplot as plt
# f, ax = plt.subplots(figsize=(15, 12))
# plt.xticks(rotation='vertical')
# import seaborn as sns
# sns.barplot(x=train_na.index, y=train_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by feature', fontsize=15)
# plt.show()

train = pd.get_dummies(train)


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=1600, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lamba=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=1500,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# 1600 2200 1500 -> 6.4

# score = rmsle_cv(lasso)
# print(f"{score.mean()} Lasso score, {score.std()}")
#
# score = rmsle_cv(ENet)
# print(f"{score.mean()} ENet score, {score.std()}")
#
# score = rmsle_cv(KRR)
# print(f"{score.mean()} KRR score, {score.std()}")
#
# score = rmsle_cv(GBoost)
# print(f"{score.mean()} GBoost score, {score.std()}")
#
# score = rmsle_cv(model_xgb)
# print(f"{score.mean()} model_xgb score, {score.std()}")
#
# score = rmsle_cv(model_lgb)
# print(f"{score.mean()} model_lgb score, {score.std()}")

"""
0.04063301125462628 Lasso score, 0.007292862859529221
0.0371175092394103 ENet score, 0.006274805307977306
10.695690183793955 KRR score, 0.7621814435523303
4.564572370854473 GBoost score, 0.22481245195491653
1.4923470302289574 model_xgb score, 0.205556038461826
2.2237074430428394 model_lgb score, 0.14684130958224545
"""

averaged_models = AveragingModels(models=(GBoost, KRR, model_lgb, lasso))

score = rmsle_cv(averaged_models)
print(f"{score.mean()} averaged_models score, {score.std()}")


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


test_id = test['subject_ID']
test.drop(['subject_ID'], axis=1, inplace=True)
train.drop(['Age'], axis=1, inplace=True)
averaged_models.fit(train.values, y_train)
test = pd.get_dummies(test)

# print(train.columns)
# print(test.columns)
train_pred = averaged_models.predict(train.values)
pred = averaged_models.predict(test.values)
print(rmsle(y_train, train_pred))


sub = pd.DataFrame()
sub['subject_ID'] = test_id
sub['年龄'] = pred.astype(int)
sub.to_csv('submission.csv', index=False)
