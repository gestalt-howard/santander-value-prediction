# Script for converting datasets into h5 versions
# Load libraries
import os
import h5py
import shutil
import pickle
import pandas as pd


# Helper functions
def save_as_h5py(fname, data):
    data.drop(columns=['ID'], inplace=True)
    with h5py.File(fname, 'w') as handle:
        handle.create_dataset('data', data=data)
    return None


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
    print 'Removing old compressed folder...'
    shutil.rmtree(compressed_path)
    print 'Creating new compressed folder...'
    os.mkdir(compressed_path)
else:
    print 'Creating new compressed folder...'
    os.mkdir(compressed_path)


# Save debug and full test sets' ID values
print 'Saving debug and full test set ID values...'
debug_id_path = compressed_path + 'debug_test_id.pickle'
full_id_path = compressed_path + 'full_test_id.pickle'
with open(debug_id_path, 'wb') as handle:
    pickle.dump(debug_test['ID'], handle)
with open(full_id_path, 'wb') as handle:
    pickle.dump(full_test['ID'], handle)


# Save compressed datasets
print 'Saving datasets...'
save_as_h5py(compressed_path + 'debug_train.h5', debug_train)
save_as_h5py(compressed_path + 'debug_test.h5', debug_test)
save_as_h5py(compressed_path + 'full_train.h5', full_train)
save_as_h5py(compressed_path + 'full_test.h5', full_test)
