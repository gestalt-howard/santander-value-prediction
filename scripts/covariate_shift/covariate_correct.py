# Script for correcting covariate shift using KLIEP

# Load libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle as pkl

import os
import pdb
import h5py
import time
import shutil

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


# High-level parameters
debug=True
random_state=0


# Helper functions and classes
# Class for identifying and removing duplicate columns
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

# Function for loading h5py file
def load_h5py(fname):
    with h5py.File(fname, 'r') as handle:
        return handle['data'][:]

# Function for loading pickle file
def load_pickle(fname):
    with open(fname, 'rb') as handle:
        return pkl.load(handle)

# Function for saving pickle file
def save_pickle(fname, data):
    with open(fname, 'wb') as handle:
        pkl.dump(obj=data, file=handle, protocol=pkl.HIGHEST_PROTOCOL)
    return None

# Function for setting up
def get_input(debug=False):
    '''
    Function for loading either debug or full datasets
    '''
    if debug:
        print 'Loading debug train and test datasets...'
        train = load_h5py('../../data/compressed/debug_train.h5')
        test = load_h5py('../../data/compressed/debug_test.h5')
        id_test = load_pickle('../../data/compressed/debug_test_id.pickle')
        y_train_log = np.log1p(load_pickle('../../data/compressed/debug_target.pickle'))
    else:
        print 'Loading original train and test datasets...'
        train = load_h5py('../../data/compressed/full_train.h5')
        test = load_h5py('../../data/compressed/full_test.h5')
        id_test = load_pickle('../../data/compressed/full_test_id.pickle')
        y_train_log = np.log1p(load_pickle('../../data/compressed/full_target.pickle'))
    # Find shape of loaded datasets
    print('Shape of training dataset: {} Rows, {} Columns'.format(*train.shape))
    print('Shape of test dataset: {} Rows, {} Columns'.format(*test.shape))
    return train, y_train_log, test, id_test

# Function for calculating Gaussian kernel value
def calc_gaussian(x, center, width):
    return np.exp(-(np.square(np.linalg.norm(np.subtract(x, center))))/(2*np.square(width)))

# Function for calculating importance weights
def get_importances(data, alpha, kc, kw):
    importance_weights = np.zeros(len(data))
    for i, row in enumerate(data):
        kernel_sum = 0
        for j, center in enumerate(kc):
            kernel_sum += alpha[j]*calc_gaussian(row, center, kw)
        importance_weights[i] = kernel_sum
    return importance_weights

# Kullback-Leibler Importance Estimation Procedure training function
def train_KLIEP(train, test, num_kernels=100, kernel_width=10, lr=0.001, a_val=1, stop=0.00001):
    '''
    Function for getting KLIEP weights for a given training and test set
    '''
    # Instantiate kernel centers
    kernel_idx_bag = np.random.permutation(len(test))
    kernel_idx = np.array([np.random.choice(kernel_idx_bag) for i in range(num_kernels)])
    kernel_centers = test[kernel_idx, :]
    # Compute A matrix
    A = np.zeros(shape=(len(test), len(kernel_centers)))
    for i, row in enumerate(test):
        for j, center in enumerate(kernel_centers):
            A[i, j] = calc_gaussian(row, center, kernel_width)
    # Compute b vector
    b = np.zeros(num_kernels)
    for j, center in enumerate(kernel_centers):
        temp_sum = 0
        for row in train:
            temp_sum += calc_gaussian(row, center, kernel_width)
        b[j] = temp_sum/np.float16(len(train))
    # Initialize alpha vector
    alpha = a_val * np.ones(shape=num_kernels)
    # Begin training
    alpha_old = np.zeros(shape=num_kernels)
    counter = 0
    while True:
        alpha = np.add(alpha, lr*np.matmul(A.T, np.divide(np.ones(len(test)), np.matmul(A, alpha))))
        alpha = np.add(alpha, np.divide(np.multiply((1-np.dot(b, alpha)), b), np.dot(b, b)))
        alpha = np.maximum(np.zeros(num_kernels), alpha)
        alpha = np.divide(alpha, np.dot(b, alpha))
        # Check convergence by average deviation
        deviation = np.linalg.norm(np.subtract(alpha, alpha_old))
        if deviation < stop*np.linalg.norm(alpha_old):
            print 'Converged in %s iterations!'%counter
            importance_weights = get_importances(data=test, alpha=alpha, kc=kernel_centers, kw=kernel_width)
            return importance_weights, alpha, kernel_centers, counter
            break
        else:
            counter += 1
            alpha_old = alpha

# Function for getting the best model
def get_best_KLIEP(train, test, width_list, path, n_splits=10, num_kernels=100, lr=0.001, a_val=1, stop=0.00001):
    '''
    Function for tuning KLIEP kernel performance
    '''
    # Split test set into disjoint subsets
    split_sets = []
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=random_state)
    print 'Splitting test set into %s disjoint subsets...'%n_splits
    for _, test_idx in kf.split(test):
        split_sets.append(test[test_idx, :])
    # Evaluate each model
    j_models = []
    alpha_list = []
    centers_list = []
    counts = []
    for idx, w in enumerate(width_list):
        print '\nWorking on split set %s'%idx
        print 'Evaluating KLIEP model with Gaussian kernel width of %s...'%w
        j_avglist = []
        counter_list = []
        for s in split_sets:
            importance, alpha, center, count = train_KLIEP(train=train, test=s, num_kernels=num_kernels,
                                                           kernel_width=w, lr=lr, a_val=a_val, stop=stop)
            j_avglist.append(np.mean(np.log(importance)))
            counter_list.append(count)
        j_models.append(np.mean(j_avglist))
        alpha_list.append(alpha)
        centers_list.append(center)
        counts.append(counter_list)
    # Evalulate train set KLIEP importances for all models
    eval_results = []
    for i, idx in enumerate((-np.array(j_models)).argsort()):
        importance_weights = get_importances(data=train, alpha=alpha_list[idx], kc=centers_list[idx],
                                             kw=width_list[idx])
        eval_results.append({'width': width_list[idx],
                             'num_kernels': num_kernels,
                             'j_value': j_models[idx],
                             'place': i,
                             'counts': counts[idx],
                             'weights': importance_weights})
    # Save train set KLIEP importances for all models
    for result in eval_results:
        save_pickle(fname=path+'%s_width%s_numk%s.pickle'%(result['place'], result['width'], result['num_kernels']),
                    data=result)

    # Return information on best model
    print '\nBest width was: %s'%eval_results[0]['width']
    return None

# Main script
def main():
    # Make covariate shift weights storage folder
    if debug:
        cs_path = './debug_cs_weights_v1/'
    else:
        cs_path = './full_cs_weights_v1/'
    if os.path.exists(cs_path):
        print 'Removing old covariate shift weights folder'
        shutil.rmtree(cs_path)
        print 'Creating new covariate shift weights folder\n'
        os.mkdir(cs_path)
    else:
        print 'Creating new covariate shift weights folder\n'
        os.mkdir(cs_path)

    # Load data
    xtrain, ytrain_log, xtest, id_test = get_input(debug)
    # Define width list
    if debug:
        wlist = [75, 500]
    else:
        wlist = [75, 100, 110, 120, 130, 140, 150, 200, 250, 300]
    # Remove duplicate columns
    unique = UniqueTransformer()
    unique.fit(X=xtrain)
    xtrain = unique.transform(X=xtrain)
    xtest = unique.transform(X=xtest)
    # Apply PCA (if necessary) and scale data
    max_features = 100
    xdata = np.concatenate([xtrain, xtest], axis=0)
    if xdata.shape[1] > max_features:
        pca = PCA(n_components=max_features)
        xdata = pca.fit_transform(xdata)
    scaler = StandardScaler()
    xdata_scaled = scaler.fit_transform(X=xdata)
    xtrain_scaled = xdata_scaled[:len(xtrain), :]
    xtest_scaled = xdata_scaled[len(xtrain):, :]

    # Define number of kernels
    if debug:
        num_kernels_list = [100, 1000]
    else:
        num_kernels_list = [100, 250, 500, 750, 1000]
    # Start training
    print 'Training KLIEP models...'
    for nk in num_kernels_list:
        print '\nEvaluating with number of kernels: %s'%nk
        get_best_KLIEP(train=xtrain_scaled,
                       test=xtest_scaled,
                       width_list=wlist,
                       num_kernels=nk,
                       path=cs_path)
    print 'KLIEP models trained and weights are saved!'


if __name__=='__main__':
    main()
