# Load libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import pdb
import os
import h5py
import pickle


# Function for loading h5py file
def load_h5py(fname):
    with h5py.File(fname, 'r') as handle:
        return handle['data'][:]
# Function for loading pickle file
def load_pickle(fname):
    with open(fname, 'rb') as handle:
        return pickle.load(handle)


# Function for setting up
def get_input(debug=False):
    '''
    Function for loading either debug or full datasets
    '''
    os.chdir('../../data/compressed/')
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
    os.chdir('../../scripts/time_series/')
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


def samples_in_batches(trn, f):
    set_size = 100
    threshold = 50

    # Row non-zero counts without target and ID
    nz_row_count = np.count_nonzero(trn[f], axis=1)
    filtered_len = len(np.where(nz_row_count>=threshold)[0])

    row_idx = np.argsort(-nz_row_count)

    counter = 0
    sample_sets = []
    while set_size < filtered_len:
        row_set_idx = row_idx[counter*set_size:(counter+1)*set_size]
        tmp_trn = trn.iloc[row_set_idx]
        tmp_trn['row_nz'] = nz_row_count[row_set_idx]
        sample_sets.append(tmp_trn)
        filtered_len -= set_size
        counter += 1

    return sample_sets

def calc_prob(c_row, df):
    '''
    Calculates probability that a row in a dataframe is part of a sequence
    belonging to c_row by the value of (intersection / union)
    '''
    # Get count information for current row
    c_vals, c_counts = np.unique(c_row, return_counts=True)
    c_dict = {'vals': c_vals, 'c_counts': c_counts}
    c_df = pd.DataFrame.from_dict(c_dict, orient='columns').set_index('vals')

    # Get count information for rows of dataframe
    probs = np.zeros(df.shape[0])
    for i, (_, row) in enumerate(df.iterrows()):
        row_vals, row_counts = np.unique(row, return_counts=True)
        row_dict = {'vals': row_vals, 'row_counts': row_counts}
        row_df = pd.DataFrame.from_dict(row_dict, orient='columns').set_index('vals')
        joint_df = c_df.join(row_df, how='outer').reset_index().replace(np.nan, 0)
        joint_df.drop([0], inplace=True)
        joint_df['abs_diff'] = np.abs(joint_df['c_counts'] - joint_df['row_counts'])
        probs[i] = joint_df['c_counts'].sum() / (joint_df['c_counts'].sum() + joint_df['abs_diff'].sum())
    return probs

def initialize_by_target(df, f):
    '''
    Chooses row with target that has highest occurrences amongst
    dataframe rows
    '''
    target_occurrences = np.zeros(df.shape[0])
    target_idx = np.zeros(df.shape[0])

    for i, (idx, row) in enumerate(df.iterrows()):
        contain_idx = (df[f]==row['target']).any(axis=1)
        target_occurrences[i] = contain_idx.sum()
        target_idx[i] = idx

    sorted_occ = np.argsort(-target_occurrences)
    best_row_idx = target_idx[sorted_occ][:10]

    return best_row_idx.astype(int)

def filter_rows(df, target):
    '''
    Filters dataframe for rows containing target
    '''
    contain_idx = (df==target).any(axis=1)
    true_idx = contain_idx[contain_idx==True].index.values
    return true_idx


def order_dataset(data, f, idx_to_use, density_df):
    '''
    Mines for structure in given data sample
    '''
    # Initialize parameters
    ignore_mask = np.ones(data.shape[0], dtype=bool)
    f_list = []
    r_list = []
    idx_list = data.index.values

    # Initialize search set
    current_idx = idx_to_use

    current_row = data[f].loc[current_idx]
    current_target = data['target'].loc[current_idx]

    while True:
        # Update ignore mask
        print 'Working on index:', current_idx
        ignore_mask[np.where(idx_list==current_idx)[0]] = False
        masked_idx = idx_list[ignore_mask]

        # Get filtered dataset
        filtered_idx = filter_rows(data[f].loc[masked_idx], current_target)
        if len(filtered_idx)==0:
            print 'No more rows are available for filtering. Exiting...'
            break
        subset_df = data[f].loc[filtered_idx]
        subset_idx = subset_df.index.values

        probs = calc_prob(current_row, subset_df)

        new_idx = subset_idx[np.argmax(probs)]
        new_row = subset_df.loc[new_idx]
        feature_mask = new_row==current_target
        new_feature = feature_mask[feature_mask==True].index.values
        # Prioritize feature if necessary (more than 1 candidate)
        if len(new_feature)>1:
            f_list.append(density_df.loc[new_feature].sort_values(by=['density']).index[-1])
        else:
            f_list.append(new_feature[0])
        # Add new row
        r_list.append(current_idx)

        # Set next iteration parameters
        current_idx = new_idx
        current_row = new_row
        current_target = data['target'].loc[current_idx]

    return f_list, r_list


# Main Script
try:
    del fnames, train, test
    print 'Clearing loaded dataframes from memory...\n'
except:
    pass
fnames, train, test = get_dataframes(debug=False)

# Column density
col_density = np.count_nonzero(train[fnames], axis=0)
density_dict = {'name': fnames, 'density': col_density}
density_df = pd.DataFrame.from_dict(density_dict, orient='columns').set_index('name')

# Get samples
samples = samples_in_batches(train, fnames)

# Get orderings
results = {}

for i, sample in enumerate(samples):
    print 'Working with sample:', i
    nz_min = np.min(sample['row_nz'])
    nz_max = np.max(sample['row_nz'])
    c = 'Sample_%s_%s'%(nz_max, nz_min)

    starting_idx_list = initialize_by_target(sample, fnames)

    sample_lists = {}
    for s_idx in starting_idx_list:
        s = 'index_%s'%s_idx
        f_list, r_list = order_dataset(sample, fnames, s_idx, density_df)
        s_dict = {'features': f_list, 'rows': r_list}
        s_df = pd.DataFrame.from_dict(s_dict, orient='columns')
        sample_lists[s] = s_df

    results[c] = sample_lists

# Save results
results_name = './candidate_sets.pickle'
with open(results_name, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
