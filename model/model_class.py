# coding=utf8
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, n_folds=5):
        self.models = models
        self.n_folds = n_folds

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        for model in self.models_:
            # for train_idx, holdout_idx in kfold.split(X, y):
            #     instance = clone(model)
            #     instance.fit(X[train_idx], y[train_idx])
            #     y_pred = instance.predict(X[holdout_idx])
            #
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])

        return np.mean(predictions, axis=1)


class StackingAverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)
