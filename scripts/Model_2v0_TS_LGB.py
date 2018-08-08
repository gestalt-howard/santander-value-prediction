# Script for running Model 2v0 via Python

# Load libraries
import warnings
warnings.filterwarnings('ignore')

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
get_scores = True


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
        pickle.dump(data, fname, protocol=pickle.HIGHEST_PROTOCOL)
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

# Make separate train set
data_train = train.copy()
data_train.drop(labels=['ID', 'target'], axis=1, inplace=True)
# Isolate and format target
target = np.log1p(train['target'].values)

# Feature Scoring using XGBoost with Leak Feature:
# Function for finding feature scores
def feature_score(num_splits=5):
    # Initialize XGBRegressor object
    reg = xgb.XGBRegressor(n_estimators=1000)

    folds = KFold(n_splits=num_splits, shuffle=True, random_state=0)
    fold_idx = [(trn, val) for trn, val in folds.split(data_train)]

    scores = []

    for idx, f in enumerate(fnames):
        feat_set = ['log_leak', f]
        score = 0
        for trn, val in fold_idx:
            reg.fit(X = data_train[feat_set].iloc[trn],
                    y = target[trn],
                    eval_set = [(data_train[feat_set].iloc[val], target[val])],
                    eval_metric = 'rmse',
                    early_stopping_rounds = 50,
                    verbose=False)
            score += rmse(target[val], reg.predict(data=data_train[feat_set].iloc[val],
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
