# Script for correcting covariate shift using KLIEP

# Load libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle as pkl

import pdb
import time

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


# High-level parameters
debug=False
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

# Function for setting up
def get_input(debug=False):
    '''
    Function for loading either debug or full datasets
    '''
    if debug:
        print 'Loading debug train and test datasets...'
        train = pd.read_csv('../../data/train_debug.csv')
        test = pd.read_csv('../../data/test_debug.csv')
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

# Function for retrieving width list
def get_width(debug=False):
    '''
    Function for loading either debug or full width lists
    '''
    if debug:
        return [10, 1000]
    else:
        return [0.1, 1, 10, 100, 1000, 10000, 100000]

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
            return importance_weights, alpha, kernel_centers
            break
        else:
            counter += 1
            alpha_old = alpha

# Function for getting the best model
def get_best_KLIEP(train, test, width_list, n_splits=10, num_kernels=100, lr=0.001, a_val=1, stop=0.00001):
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
    for idx, w in enumerate(width_list):
        print '\nWorking on split set %s'%idx
        print 'Evaluating KLIEP model with Gaussian kernel width of %s...'%w
        j_avglist = []
        for s in split_sets:
            importance, alpha, center = train_KLIEP(train=train, test=s, num_kernels=num_kernels, kernel_width=w,
                                     lr=lr, a_val=a_val, stop=stop)
            j_avglist.append(np.mean(np.log(importance)))
        j_models.append(np.mean(j_avglist))
        alpha_list.append(alpha)
        centers_list.append(center)
    # Use best model to evaluate train set KLIEP importances
    best_idx = np.argmax(np.array(j_models))
    print '\nBest width was: %s'%width_list[best_idx]
    importance_weights = get_importances(data=train, alpha=alpha_list[best_idx], kc=centers_list[best_idx],
                                         kw=width_list[best_idx])
    return width_list[best_idx], importance_weights


# Main script
def main():
    # Load data
    xtrain, ytrain_log, xtest, id_test = get_input(debug)
    # Load width list
    wlist = get_width(debug)
    # Remove duplicate columns
    unique = UniqueTransformer()
    unique.fit(X=xtrain)
    xtrain = unique.transform(X=xtrain)
    xtest = unique.transform(X=xtest)
    # Scale data
    xdata = np.concatenate([xtrain, xtest], axis=0)
    scaler = StandardScaler()
    xdata_scaled = scaler.fit_transform(X=xdata)
    xtrain_scaled = xdata_scaled[:len(xtrain), :]
    xtest_scaled = xdata_scaled[len(xtrain):, :]

    # Get KLIEP weights
    best_width, importances = get_best_KLIEP(xtrain_scaled, xtest_scaled, wlist)
    # Save KLIEP results
    print 'Saving trained importance weights...'
    save_name = 'kliep_weights.pickle'
    with open(save_name, 'wb') as handle:
        pkl.dump(obj=(best_width, importances), file=handle, protocol=pkl.HIGHEST_PROTOCOL)
    print 'Importance weights saved!'


if __name__=='__main__':
    main()
