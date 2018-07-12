# Python script version of iPython notebook
# XGBoost Regressor with Pipelined Feature Engineering

# Load libraries
import numpy as np
import pandas as pd
import xgboost as xgb

import dill
import pdb

from scipy.stats import skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin  # For making custom classes
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import KFold

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

# High-level parameters
debug=False
random_state=0

# Helper functions and classes
class UniqueTransformer(BaseEstimator, TransformerMixin):
    '''
    Class with fit and transform methods for removing duplicate columns from a dataset
    **fit** finds the indexes of unique columns using numpy unique
    **transform** returns the dataset with the indexes of unique columns
    '''
    def __init__(self, axis=1):
        if axis==0:
            raise NotImplementedError('Axis is 0! Not implemented!')
        self.axis=axis

    def fit(self, X, y=None):
        print 'Finding unique indexes...'
        _, self.unique_indexes_ = np.unique(X, axis=self.axis, return_index=True)
        return self

    def transform(self, X, y=None):
        print 'Filtering for only unique columns...'
        return X[:, self.unique_indexes_]


class ClassifierTransformer(BaseEstimator, TransformerMixin):
    '''
    Class describing an object that transforms datasets via estimator results
    **_get_labels** specifies target value bins and transforms target vector into bin values
    '''
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator=estimator
        self.n_classes=n_classes
        self.cv=cv

    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us)/self.n_classes)

        for i_class in range(self.n_classes):
            if i_class+1 == self.n_classes:  # Edge case where i_class is initialized at 1
                y_labels[y >= y_us[i_class*step]] = i_class
            else:
                y_labels[np.logical_and(y>=y_us[i_class*step], y<y_us[(i_class+1)*step])] = i_class
        return y_labels

    def fit(self, X, y):
        print 'Fitting random forest classifier with n_classes = %s'%self.n_classes
        y_labels = self._get_labels(y)
        kf = KFold(n_splits=self.cv, shuffle=False, random_state=random_state)
        self.estimators_ = []
        # Train individual classifiers
        for train, _ in kf.split(X, y_labels):
            self.estimators_.append(clone(self.estimator).fit(X[train], y_labels[train]))
        return self

    def transform(self, X, y=None):
        print 'Applying classifier transformation with n_classes = %s'%self.n_classes
        kf = KFold(n_splits=self.cv, shuffle=False, random_state=random_state)

        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])

        for estimator, (_, test) in zip(self.estimators_, kf.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])


# Function for transforming a row into statistical values
def apply_stats_to_row(row):
    stats = []
    for fun in stat_functions:
        stats.append(fun(row))
    return stats

class StatsTransformer(BaseEstimator, TransformerMixin):
    '''
    Class describing an object for transforming datasets into statistical values row-wise
    NOTE: This class is dependent on the function **apply_stats_to_row**
    '''
    def __init__(self, verbose=0, n_jobs=-1, pre_dispatch='2*n_jobs'):
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print 'Applying statistical transformation to dataset...'
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, verbose=self.verbose)
        # Get statistics transformation
        stats_list = parallel(delayed(apply_stats_to_row)(X[i_smpl, :]) for i_smpl in range(len(X)))
        return np.array(stats_list)


class XGBRegressorCV(BaseEstimator, RegressorMixin):
    '''
    Class that describes an object that implements XGB Regressor with cross-validation capability
    **fit** defines indexes for cross-validation purposes and fits individual XGBRegressor models
    '''
    def __init__(self, xgb_params=None, fit_params=None, cv=3):
        self.xgb_params = xgb_params
        self.fit_params = fit_params
        self.cv = cv

    @property
    def feature_importances_(self):
        feature_importances = []
        for estimator in self.estimators_:
            feature_importances.append(estimator.feature_importances_)
        return np.mean(feature_importances, axis=0)

    @property
    def evals_result_(self):
        evals_result = []
        for estimator in self.estimators_:
            evals_result.append(estimator.evals_result_)
        return np.array(evals_result)

    @property
    def best_scores_(self):
        best_scores = []
        for estimator in self.estimators_:
            best_scores.append(estimator.best_score)
        return np.array(best_scores)

    @property
    def best_score_(self):
        return np.mean(self.best_scores_)

    @property
    def best_iterations_(self):
        best_iterations = []
        for estimator in self.estimators_:
            best_iterations.append(estimator.best_iteration)
        return np.array(best_iterations)

    @property
    def best_iteration_(self):
        return np.round(np.mean(self.best_iterations_))

    def fit(self, X, y, **fit_params):
        print 'Fitting XGB Regressors...'
        kf = KFold(n_splits=self.cv, shuffle=False, random_state=random_state)
        self.estimators_ = []

        for train, valid in kf.split(X, y):
            self.estimators_.append(xgb.XGBRegressor(**self.xgb_params).fit(X[train], y[train],
                                                                            eval_set=[(X[valid], y[valid])],
                                                                            **self.fit_params))
        return self

    def predict(self, X):
        print 'Making aggregate XGB Regressor predictions...'
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))
        return np.mean(y_pred, axis=0)


class _StatFunAdaptor:
    '''
    Class describing an object that wraps pre-processing functions with a main statistical function
    **__init__** sets up the object parameters
    **__call__** describes routine steps when object is called
    '''
    def __init__(self, stat_fun, *funs, **stat_fun_kwargs):
        self.stat_fun = stat_fun
        self.funs = funs
        self.stat_fun_kwargs = stat_fun_kwargs

    def __call__(self, x):
        x = x[x != 0]  # Only look at nonzero entries
        # Transform row with cached functions
        for fun in self.funs:
            x = fun(x)
        if x.size == 0:
            return -99999  # Edge case default
        return self.stat_fun(x, **self.stat_fun_kwargs)  # Returns result of a run


def diff2(x):
    return np.diff(x, n=2)


def get_stat_funs():
    '''
    Function for defining all the statistical functions used for evaluating elements in a row-wise manner
    Functions include: length, minimum, maximum, standard deviation, skew, kurtosis, and percentile
    '''
    stat_funs = []

    stats = [len, np.min, np.max, np.median, np.std, skew, kurtosis] + 19 * [np.percentile]
    # Dictionary arguments (nontrivial only for percentile function)
    stats_kwargs = [{} for i in range(7)] + [{'q': i} for i in np.linspace(0.05, 0.95, 19)]

    for stat, stat_kwargs in zip(stats, stats_kwargs):
        stat_funs.append(_StatFunAdaptor(stat,**stat_kwargs))
        stat_funs.append(_StatFunAdaptor(stat, np.diff, **stat_kwargs))  # Apply to 1-diff of row
        stat_funs.append(_StatFunAdaptor(stat, diff2, **stat_kwargs))  # Apply to 2-diff of row
        stat_funs.append(_StatFunAdaptor(stat, np.unique, **stat_kwargs))  # Apply to unique vals of row
        stat_funs.append(_StatFunAdaptor(stat, np.unique, np.diff, **stat_kwargs))  # Apply to unique, 1-diff row vals
        stat_funs.append(_StatFunAdaptor(stat, np.unique, diff2, **stat_kwargs))  # Apply to unique, 2-diff row vals
    return stat_funs


# Function for retrieving a Random Forest Classifier object
def get_rfc():
    return RandomForestClassifier(n_estimators=100,
                                  max_features=0.5,
                                  max_depth=None,
                                  max_leaf_nodes=270,
                                  min_impurity_decrease=0.0001,
                                  random_state=123,
                                  n_jobs=-1)


# Function for setting up datasets
def get_input(debug=False):
    if debug:
        print 'Loading debug train and test datasets...'
        train = pd.read_csv('../data/train_debug.csv')
        test = pd.read_csv('../data/test_debug.csv')
    else:
        print 'Loading original train and test datasets...'
        train = pd.read_csv('../data/train.csv')
        test = pd.read_csv('../data/test.csv')
    y_train_log = np.log1p(train['target'])
    id_test = test['ID']
    # Drop unnecessary columns
    train.drop(labels=['ID', 'target'], axis=1, inplace=True)
    test.drop(labels=['ID'], axis=1, inplace=True)
    # Find shape of loaded datasets
    print('Shape of training dataset: {} Rows, {} Columns'.format(*train.shape))
    print('Shape of test dataset: {} Rows, {} Columns'.format(*test.shape))

    return train.values, y_train_log.values, test.values, id_test.values


# Main script
def main():
    # XGBoost regressor parameters
    xgb_params = {'n_estimators': 1000,
                  'objective': 'reg:linear',
                  'booster': 'gbtree',
                  'learning_rate': 0.02,
                  'max_depth': 22,
                  'min_child_weight': 57,
                  'gamma' : 1.45,
                  'alpha': 0.0,
                  'lambda': 0.0,
                  'subsample': 0.67,
                  'colsample_bytree': 0.054,
                  'colsample_bylevel': 0.50,
                  'n_jobs': -1,
                  'random_state': 456}
    # Fitting XGB Regressor parameters
    fit_params = {'early_stopping_rounds': 15,
                  'eval_metric': 'rmse',
                  'verbose': False}
    # Define pipeline
    pipe = Pipeline([
        ('vt', VarianceThreshold(threshold=0.0)),
        ('ut', UniqueTransformer()),
        ('fu', FeatureUnion([
            ('pca', PCA(n_components=100)),
            ('ct-2', ClassifierTransformer(get_rfc(), n_classes=2, cv=5)),
            ('ct-3', ClassifierTransformer(get_rfc(), n_classes=3, cv=5)),
            ('ct-4', ClassifierTransformer(get_rfc(), n_classes=4, cv=5)),
            ('ct-5', ClassifierTransformer(get_rfc(), n_classes=5, cv=5)),
            ('st', StatsTransformer(verbose=2))
        ])),
        ('xgb-cv', XGBRegressorCV(xgb_params=xgb_params, fit_params=fit_params, cv=10))
    ])
    # Get data
    X_train, y_train_log, X_test, id_test = get_input(debug)
    # Train pipeline
    pipe.fit(X_train, y_train_log)
    print 'Best CV scores:\n', pipe.named_steps['xgb-cv'].best_scores_
    print 'Averged CV scores:\n', pipe.named_steps['xgb-cv'].best_score_
    # Get predictions
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    # Format submissions
    submission_path = '../submissions/xgbpipe_0v2_submit.csv'
    submission = pd.DataFrame()
    submission['ID'] = id_test
    submission['target'] = y_pred
    # Save submissions
    submission.to_csv(submission_path, index=False)


if __name__=='__main__':
    # Define stat functions
    stat_functions = get_stat_funs()
    main()
