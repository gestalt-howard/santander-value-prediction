# Script for converting datasets into h5 versions
# Load libraries
import os
import h5py
import shutil
import pickle
import pandas as pd


# Helper functions
# Function for saving datasets in h5py format
def save_as_h5py(fname, data):
    data.drop(columns=['ID'], inplace=True)
    with h5py.File(fname, 'w') as handle:
        handle.create_dataset('data', data=data)
    return None

# Function for saving data as pickle files
def save_as_pickle(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Load data
data_path = '../data/'
# Debug set
print 'Loading debug datasets...'
debug_train = pd.read_csv(data_path + 'train_debug.csv', index_col=0)
debug_test = pd.read_csv(data_path + 'test_debug.csv', index_col=0)
# Full set
print 'Loading full datasets...'
full_train = pd.read_csv(data_path + 'train.csv')
full_test = pd.read_csv(data_path + 'test.csv')


# Make folder for compressed folder
compressed_path = '../data/compressed/'
if os.path.exists(compressed_path):
    print '\nRemoving old compressed folder...'
    shutil.rmtree(compressed_path)
    print 'Creating new compressed folder...'
    os.mkdir(compressed_path)
else:
    print '\nCreating new compressed folder...'
    os.mkdir(compressed_path)


# Save debug and full test sets' ID values
print '\nSaving debug and full test set ID values...'
debug_id_path = compressed_path + 'debug_test_id.pickle'
full_id_path = compressed_path + 'full_test_id.pickle'
with open(debug_id_path, 'wb') as handle:
    pickle.dump(debug_test['ID'], handle)
with open(full_id_path, 'wb') as handle:
    pickle.dump(full_test['ID'], handle)


# Save debug and full test sets' ID values
print '\nSaving debug and full train set ID values...'
debug_trid_path = compressed_path + 'debug_train_id.pickle'
full_trid_path = compressed_path + 'full_train_id.pickle'
save_as_pickle(debug_trid_path, debug_train['ID'])
save_as_pickle(full_trid_path, full_train['ID'])


# Save debug and full train sets' target values
print '\nSaving debug and full train set target values...'
debug_target_path = compressed_path + 'debug_target.pickle'
full_target_path = compressed_path + 'full_target.pickle'
with open(debug_target_path, 'wb') as handle:
    pickle.dump(debug_train['target'], handle)
with open(full_target_path, 'wb') as handle:
    pickle.dump(full_train['target'], handle)
print 'Removing target column for debug and full train sets...'
debug_train.drop(columns=['target'], inplace=True)
full_train.drop(columns=['target'], inplace=True)


# Save feature names
print '\nSaving feature names...'
features_path = compressed_path + 'feature_names.pickle'
feature_names = debug_test.loc[:, debug_test.columns!='ID'].columns.values
save_as_pickle(features_path, feature_names)


# Save debug and full train and test sets' indexes
print '\nSaving debug and full train and test sets indexes'
debug_trainidx_path = compressed_path + 'debug_trainidx.pickle'
debug_testidx_path = compressed_path + 'debug_testidx.pickle'
full_trainidx_path = compressed_path + 'full_trainidx.pickle'
full_testidx_path = compressed_path + 'full_testidx.pickle'
save_as_pickle(debug_trainidx_path, debug_train.index)
save_as_pickle(debug_testidx_path, debug_test.index)
save_as_pickle(full_trainidx_path, full_train.index)
save_as_pickle(full_testidx_path, full_test.index)


# Save compressed datasets
print '\nSaving datasets...'
save_as_h5py(compressed_path + 'debug_train.h5', debug_train)
save_as_h5py(compressed_path + 'debug_test.h5', debug_test)
save_as_h5py(compressed_path + 'full_train.h5', full_train)
save_as_h5py(compressed_path + 'full_test.h5', full_test)
