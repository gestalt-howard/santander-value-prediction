# Script for generating feature scores
# Load libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt

import pdb
import os
import gc; gc.enable()
import h5py
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Function for loading h5py file
def load_h5py(fname):
    with h5py.File(fname, 'r') as handle:
        return handle['data'][:]
# Function for loading pickle file
def load_pickle(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


# Function for setting up
def get_input():
    '''
    Function for loading full datasets
    '''
    os.chdir('../data/compressed/')
    print os.getcwd()
    pkl_files = ['train_id.pickle', 'trainidx.pickle', 'target.pickle', 'test_id.pickle', 'testidx.pickle']

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
def get_dataframes():
    # Load data
    fnames, train, id_train, train_idx, target, test, id_test, test_idx = get_input()
    # Format data
    train_df = pd.DataFrame(data=train, index=train_idx, columns=fnames)
    train_df['ID'] = id_train
    train_df['target'] = target
    test_df = pd.DataFrame(data=test, index=test_idx, columns=fnames)
    test_df['ID'] = id_test

    print('\nShape of training dataframe: {} Rows, {} Columns'.format(*train_df.shape))
    print('Shape of test dataframe: {} Rows, {} Columns'.format(*test_df.shape))
    return fnames, train_df, test_df


# Function for loading leaks
def load_leaks(leak_val):
    leak_dir = './time_series/stats/'

    train_leak_loc = leak_dir + 'train_leak_%s.csv'%leak_val
    train_leak = pd.read_csv(train_leak_loc).compiled_leak
    test_leak_loc = leak_dir + 'test_leak_%s.csv'%leak_val
    test_leak = pd.read_csv(test_leak_loc).compiled_leak

    return train_leak, test_leak


# Function for calculating ROOT mean squared error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Function for finding feature scores
def feature_score(train, target, fnames, num_splits=5):
    # Initialize XGBRegressor object
    reg = xgb.XGBRegressor(n_estimators=1000)

    folds = KFold(n_splits=num_splits, shuffle=True, random_state=0)
    fold_idx = [(trn, val) for trn, val in folds.split(train)]

    scores = []

    for idx, f in enumerate(fnames):
        print 'Scoring feature %s/%s'%(idx+1, len(fnames))
        feat_set = ['log_leak', f]
        score = 0
        for trn, val in fold_idx:
            reg.fit(X = train[feat_set].iloc[trn],
                    y = target[trn],
                    eval_set = [(train[feat_set].iloc[val], target[val])],
                    eval_metric = 'rmse',
                    early_stopping_rounds = 50,
                    verbose=False)
            score += rmse(target[val], reg.predict(data=train[feat_set].iloc[val],
                                                   ntree_limit=reg.best_ntree_limit)) / folds.n_splits
        scores.append((f, score))
    return scores


# Function for feature engineering
def get_scores(train, target, f, leak_train, lagval):
    '''
    - Formats train and test dataframes for training
    - Performs feature scoring using XGBoost Regressor
    - Appends training leak and test leak to respective dataframes
    '''

    tmp_trn = train.copy(deep=True)
    tmp_trn['leak'] = leak_train
    tmp_trn['log_leak'] = np.log1p(leak_train)

    score_name = './model_data/model_2v2_featscores_%s.csv'%lagval
    print '\nGenerating feature scores...'
    scores = feature_score(train=tmp_trn, target=target, fnames=f)
    score_df = pd.DataFrame(data=scores, columns=['feature', 'rmse']).set_index('feature')
    score_df.sort_values(by='rmse', ascending=True, inplace=True)
    score_df.to_csv(score_name)

    return None


# Main Script
try:
    del fnames, train, test
    print 'Clearing loaded dataframes from memory...\n'
except:
    pass
fnames, train, test = get_dataframes()

# Load leaks
leak_val=38
print '\nLoading train and test leaks...\n'
train_leak, test_leak = load_leaks(leak_val)
# Format target variable
target = train['target'].values
target_log = np.log1p(target)

# Get scores
get_scores(train, target_log, fnames, train_leak, leak_val)
