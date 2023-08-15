# coding=utf8
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from utils.save_file import save_model
import numpy as np
import pickle

# Validation function
n_folds = 10


def rmsle_cv(model, train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def mae_cv(model, train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    mae = -cross_val_score(model, train.values, y_train, scoring='neg_mean_absolute_error', cv=kf)
    return mae


def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))


def model_train(model_input, train, y_train, model_name=''):
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    scores = []
    train_scores = []
    model = model_input
    for train_idx, val_idx in kfold.split(train):
        X_train_fold, X_val_fold = train.values[train_idx], train.values[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        score = calculate_mae(y_val_fold, y_pred)
        y_train_pred = model.predict(X_train_fold)
        train_score = calculate_mae(y_train_fold, y_train_pred)

        models.append(model)
        scores.append(score)

        train_scores.append(train_score)

    mean_score = np.mean(scores)
    mean_train_score = np.mean(train_scores)

    print(model_name + '训练精度:')
    print(f'每折分数：{train_scores}')
    print(f'平均分数：{mean_train_score}')
    print(model_name + '测试精度')
    print(f'每折分数：{scores}')
    print(f'平均分数：{mean_score}')
    file_name = model_name + '.pkl'
    save_model(file_name, models)


def model_pred(test, file_name=''):
    with open(file_name, 'rb') as f:
        model_list = pickle.load(f)
    predictions = []

    for model in model_list:
        y_pred = model.predict(test.values)
        predictions.append(y_pred)

    averaged_predictions = np.mean(predictions, axis=0)
    return averaged_predictions
