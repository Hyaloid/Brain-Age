# coding=utf8
from modeling.model_class import AveragingModels, StackingAverageModels
from modeling.base_models import *
from group_lasso import GroupLasso


# stacking model
def my_stacking_model():
    stacked_averaged_model1 = StackingAverageModels(base_models=(lasso, model_lgb), meta_model=model_lgb)
    stacked_averaged_model2 = StackingAverageModels(base_models=(stacked_averaged_model1, model_lgb), meta_model=lasso)
    return stacked_averaged_model2


# 创建组套索模型
def group_lasso_model(train):  # 效果奇差无比 -> 特征之间有依赖
    pred_arr = []
    for col in train:
        if 'lh' in col or 'Left' in col:
            pred_arr.append(1)
        elif 'rh' in col or 'Right' in col:
            pred_arr.append(2)
        else:
            pred_arr.append(-1)
    gls = GroupLasso(groups=pred_arr, n_iter=50000)
    return gls


# 标准化
def my_averaged_model():
    averaged_models = AveragingModels(models=(GBoost, KRR, model_lgb, lasso))
    return averaged_models
