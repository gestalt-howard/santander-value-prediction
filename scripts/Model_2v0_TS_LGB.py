# Script for running Model 2v0 via Python

# Load libraries
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pdb
import os
import h5py
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# Debug flag
debug = False
# Get feature scores flag
get_scores = False


# Helper Functions:
# Function for loading h5py file
def load_h5py(fname):
    with h5py.File(fname, 'r') as handle:
        return handle['data'][:]
# Function for loading pickle file
def load_pickle(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)
# Function for saving pickle file
def save_pickle(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

# Function for setting up
def get_input(debug=False):
    '''
    Function for loading either debug or full datasets
    '''
    os.chdir('../data/compressed/')
    print os.getcwd()
    pkl_files = ['train_id.pickle', 'trainidx.pickle', 'target.pickle', 'test_id.pickle', 'testidx.pickle']
    if debug:
        print 'Loading debug train and test datasets...'
        # h5py files
        train = load_h5py('debug_train.h5')
        test = load_h5py('debug_test.h5')
        # pickle files
        id_train, train_idx, target, id_test, test_idx = [load_pickle('debug_%s'%f) for f in pkl_files]
    else:
        print 'Loading original train and test datasets...'
        # h5py files
        train = load_h5py('full_train.h5')
        test = load_h5py('full_test.h5')
        # pickle files
        id_train, train_idx, target, id_test, test_idx = [load_pickle('full_%s'%f) for f in pkl_files]
    # Load feature names
    fnames = load_pickle('feature_names.pickle')
    # Find shape of loaded datasets
    print('Shape of training dataset: {} Rows, {} Columns'.format(*train.shape))
    print('Shape of test dataset: {} Rows, {} Columns'.format(*test.shape))
    os.chdir('../../scripts/')
    print os.getcwd()
    return fnames, train, id_train, train_idx, target, test, id_test, test_idx

# Function for getting datasets in dataframe format
def get_dataframes(debug=False):
    # Load data
    fnames, train, id_train, train_idx, target, test, id_test, test_idx = get_input(debug)
    # Format data
    train_df = pd.DataFrame(data=train, index=train_idx, columns=fnames)
    train_df['ID'] = id_train
    train_df['target'] = target
    test_df = pd.DataFrame(data=test, index=test_idx, columns=fnames)
    test_df['ID'] = id_test

    print('\nShape of training dataframe: {} Rows, {} Columns'.format(*train_df.shape))
    print('Shape of test dataframe: {} Rows, {} Columns'.format(*test_df.shape))
    return fnames, train_df, test_df

# Function for calculating ROOT mean squared error
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**.5


# Main Script:
try:
    del fnames, train, test
    print 'Clearing loaded dataframes from memory...\n'
except:
    pass
fnames, train, test = get_dataframes(debug=debug)

# Load leak values
leak_path = './time_series/stats/'
path_train_leak = leak_path + 'train_leak.csv'
path_test_leak = leak_path + 'test_leak.csv'

# Add train leak
train_leak = pd.read_csv(path_train_leak)
train['leak'] = train_leak['compiled_leak'].replace(np.nan, 0.0)
train['log_leak'] = np.log1p(train['leak'].values)
# Add test leak
test_leak = pd.read_csv(path_test_leak)
test['leak'] = test_leak['compiled_leak'].replace(np.nan, 0.0)
test['log_leak'] = np.log1p(test['leak'].values)

# Isolate and format target
target = np.log1p(train['target'].values)

# Feature Scoring using XGBoost with Leak Feature:
# Function for finding feature scores
def feature_score(num_splits=5):
    # Initialize XGBRegressor object
    reg = xgb.XGBRegressor(n_estimators=1000)

    folds = KFold(n_splits=num_splits, shuffle=True, random_state=0)
    fold_idx = [(trn, val) for trn, val in folds.split(train)]

    scores = []

    for idx, f in enumerate(fnames):
        feat_set = ['log_leak', f]
        score = 0
        for trn, val in fold_idx:
            reg.fit(X = train[feat_set].iloc[trn],
                    y = target[trn],
                    eval_set = [(train[feat_set].iloc[val], target[val])],
                    eval_metric = 'rmse',
                    early_stopping_rounds = 50,
                    verbose=True)
            score += rmse(target[val], reg.predict(data=train[feat_set].iloc[val],
                                                   ntree_limit=reg.best_ntree_limit)) / folds.n_splits
        scores.append((f, score))

    return scores

score_name = './model_data/model_2v0_featscores.pickle'
if get_scores:
    # Get scores
    scores = feature_score(num_splits=5)
    # Save scores
    save_pickle(score_name, scores)
else:
    # Load scores
    scores = load_pickle(score_name)

# Create dataframe from scores
score_df = pd.DataFrame(data=scores, columns=['feature', 'rmse']).set_index('feature')
score_df.sort_values(by='rmse', ascending=True, inplace=True)
# Select good features
threshold = 0.7925
good_features = score_df.loc[score_df['rmse']<=threshold].index
good_rmse = score_df.loc[score_df['rmse']<=threshold, 'rmse'].values

# Train LightGBM
# Function for calculating row-wise metadata
def add_metadata(df):
    df.replace(0, np.nan, inplace=True)
    # Calculate new metadata
    df['log_of_mean'] = np.log1p(df.loc[:, fnames].mean(axis=1))
    df['mean_of_log'] = np.mean(np.log1p(df.loc[:, fnames]), axis=1)
    df['log_of_median'] = np.log1p(df.loc[:, fnames].median(axis=1))
    df['num_nans'] = df.loc[:, fnames].isnull().sum(axis=1)
    df['sum'] = df.loc[:, fnames].sum(axis=1)
    df['std'] = df.loc[:, fnames].std(axis=1)
    df['kurtosis'] = df.loc[:, fnames].kurtosis(axis=1)
    return df

# Add row-wise metadata to train and test sets
print '\nAdding metadata to train and test sets...\n'
data_train = add_metadata(train)
data_test = add_metadata(test)
# Add target column to test set
data_test['target'] = 0
# Define features to be used in training LGBM
flist = good_features.tolist() + ['log_of_mean', 'mean_of_log', 'log_of_median',
                                  'num_nans', 'sum', 'std', 'kurtosis']

# Define LGBM training set
dtrain = lgb.Dataset(data=data_train[flist],
                     label=target, free_raw_data=True)
dtrain.construct()
# Train LightGBM
lgb_params = {
        'objective': 'regression',
        'num_leaves': 58,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.05,
        'metric': 'l2',
            }

folds = KFold(n_splits=5, shuffle=True, random_state=0)
train_pred = np.zeros(data_train.shape[0])

for trn, val in folds.split(data_train):
    reg = lgb.train(params=lgb_params,
                    train_set = dtrain.subset(trn),
                    valid_sets = dtrain.subset(val),
                    num_boost_round = 10000,
                    early_stopping_rounds = 100,
                    verbose_eval = 100)

    # Get training predictions
    train_pred[val] = reg.predict(data_train[flist].iloc[val])
    # Get test predictions
    test['target'] = reg.predict(data_test[flist]) / folds.n_splits
    # Print validation error
    print 'Validation error:', mean_squared_error(train_pred[val], target[val])**.5

# Evaluate Results and Save Submission:
# Evaluate training results
data_train['predictions'] = train_pred
data_train.loc[data_train['leak'].notnull(), 'predictions'] = np.log1p(data_train.loc[data_train['leak'].notnull(),
                                                                                      'leak'])
print 'Train score:', mean_squared_error(target, train_pred)**.5
print 'Train score with leak:', mean_squared_error(target, data_train['predictions'])**.5

# Save test submission
sub_name = '../submissions/ts_lgb_2v0_submit.csv'

data_test['target'] = np.expm1(data_test['target'])
data_test.loc[data_test['leak'].notnull(), 'target'] = data_test.loc[data_test['leak'].notnull(), 'leak']
data_test[['ID', 'target']].to_csv(sub_name, index=False, float_format='%.2f')
