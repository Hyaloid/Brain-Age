from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1, max_iter=50000))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=1600, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=1)

RFReg = RandomForestRegressor(n_estimators=1600, criterion="squared_error",
                              max_depth=4, min_samples_split=5, min_samples_leaf=15,
                              warm_start=True)

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

# score = rmsle_cv(RFReg)
# print(f"{score.mean()} RFReg score, {score.std()}")

"""
0.04063301125462628 Lasso score, 0.007292862859529221
0.0371175092394103 ENet score, 0.006274805307977306
10.695690183793955 KRR score, 0.7621814435523303
4.564572370854473 GBoost score, 0.22481245195491653
1.4923470302289574 model_xgb score, 0.205556038461826
2.2237074430428394 model_lgb score, 0.14684130958224545
0.6252308321098174 RFReg score, 0.11987177896245349
"""
