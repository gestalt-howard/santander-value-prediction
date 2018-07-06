# Script for training and running deep autoencoder
# Import libraries
import warnings
warnings.filterwarnings('ignore')

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
import json
import pdb


# Helper functions:
# Keras model history class
class history(object):
    def __init__(self, fname):
        '''Defines main parameters for history object'''
        self.fname = fname
        if os.path.exists(self.fname):
            print 'Loading existing history file...'
            with open(self.fname, 'r') as handle:
                self.history = json.load(handle)
        else:
            print 'Creating new history file...'
            self.history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
            with open(self.fname, 'w') as handle:
                json.dump(self.history, handle)


    def append_history(self, new_history):
        '''Updates history object with new training history'''
        for key, value in new_history.iteritems():
            for v in value:
                self.history[key].append(v)
        print 'Saving updated history file...'
        with open(self.fname, 'w') as handle:
            json.dump(self.history, handle)


    def visualize_accuracy(self):
        '''Plots visualization of training and validation accuracies'''
        t_acc = self.history['acc']
        v_acc = self.history['val_acc']
        # Make plot
        plt.figure(figsize=(10, 7))
        plt.plot(t_acc, label='training_acc')
        plt.plot(v_acc, label='validation_acc')
        plt.legend(loc='upper left')
        plt.title('Training and Validation Accuracies vs. Epoch Count')
        plt.xlabel('Epoch Count')
        plt.show()


# Get current epoch
def get_current_epoch(n_prefix):
    files = sorted([f for f in os.listdir(weights_path) if n_prefix in f])
    if len(files) == 0:
        print 'No epochs have yet been run'
        return(-1)
    else:
        latest_idx = int(files[-1].split(weights_suffix)[0].split('_')[-1])-1
        print 'Latest epoch:', latest_idx
        return(latest_idx)


# Format current epoch
def format_epoch(enum):
    strnum = str(enum)
    fill_val = 3 - len(strnum)
    if fill_val > 0:
        return(fill_val*str(0)+strnum)
    else:
        return(strnum)


# Creating stacked autoencoders
def get_stacked_models(input_size, num_ae):
    encoded_input = Input(shape=(input_size,))
    autoencoder_list = []
    encoder_list = []
    for n in range(num_ae):
        encoding_dim = int(input_size/(2**(n+1)))  # Size of bottleneck layer
        decoding_dim = int(input_size/(2**n))  # Size of reconstruction
        # Define autoencoder layers
        encoded = Dense(encoding_dim, activation='relu', name='encode_%s'%str(n))(encoded_input)
        decoded = Dense(decoding_dim, activation='relu', name='decode_%s'%str(n))(encoded)
        # Define and compile models
        autoencoder = Model(encoded_input, decoded)
        encoder = Model(encoded_input, encoded)
        opt = Adam(lr=0.0001)
        autoencoder.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
        autoencoder_list.append(autoencoder)
        encoder_list.append(encoder)
        # Reset encoded input to smaller size
        encoded_input = Input(shape=(encoding_dim,))
    return(encoder_list, autoencoder_list)

# Main Script
# Debug flag:
debug = False

# Load data:
data_path = '../../data/'
if debug:
    print 'Using debugging datasets'
    train_path = data_path + 'train_debug.csv'
    test_path = data_path + 'test_debug.csv'
    epoch_num = 5
else:
    print 'Using full datasets'
    train_path = data_path + 'train.csv'
    test_path = data_path + 'test.csv'
    epoch_num = 400
print 'Loading training and test dataframes...'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print 'Training and test dataframes loaded!'

# Data paths
parent_path = os.getcwd() + '/'
# Autoencoder weight files metadata
weights_path = parent_path + 'weights/'
weights_prefix = 'ae_weight_'
weights_suffix = '.hdf5'
# Autoencoder transformed data metadata
trans_path = parent_path + 'data/'
trans_prefix = trans_path + 'transformed_'
trans_suffix = '.csv'
# History file metadata
history_prefix = weights_path + 'history_'

# Make folders if they don't exist
if not os.path.exists(weights_path):
    print '\nCreating weights folder...'
    os.mkdir(weights_path)
if not os.path.exists(trans_path):
    print '\nCreating transformed data folder...'
    os.mkdir(trans_path)

# Isolate training labels
labels = train_df['target']
train_df.drop(columns=['target'], inplace=True, axis=1)
print('\nShape of training dataset: {} Rows, {} Columns'.format(*train_df.shape))
print('Shape of test dataset: {} Rows, {} Columns'.format(*test_df.shape))

# Create aggregate train + test dataframe
print '\nScaling data and assembling master dataframe...'
all_data = pd.concat([train_df, test_df], axis=0)
all_data.set_index('ID', inplace=True)
# Get columns
all_cols = all_data.columns.values
all_idx = all_data.index.values
# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data)
scaled_data = pd.DataFrame(data=scaled_data, columns=all_cols, index=all_idx)
scaled_data.index.name = 'ID'
print 'Master dataframe assembled'

# Stacked autoencoder parameters
visualize_flag = False
num_ae = 5
batch_size = 100
num_train = train_df.shape[0]
input_size = all_data.shape[1]
print '\nAutoencoder input size:', input_size

# Create autoencoder and encoder models
encoder_list, autoencoder_list = get_stacked_models(input_size, num_ae)
# Train stacked autoencoders
current_data = scaled_data
history_list = []  # List of history objects
epoch_idx_params = dict()
# Iterate through stacks
for n, autoencoder in enumerate(autoencoder_list):
    print '\nTraining %s autoencoder stack...'%n
    print('Shape of current data: {} Rows, {} Columns'.format(*current_data.shape))
    n_prefix = '%s_%s'%(str(n), weights_prefix)
    # Initialize history file
    epoch_history = history(history_prefix + str(n) + '.json')
    # Load old weights (if exists)
    latest_idx = get_current_epoch(n_prefix)
    prior_weight_fname = weights_path + n_prefix + format_epoch(latest_idx) + weights_suffix
    if os.path.exists(prior_weight_fname):
        print 'Loading existing model weight...\n', prior_weight_fname
        autoencoder.load_weights(prior_weight_fname, by_name=True)
    # Define performance tracking template
    weight_fname = weights_path + n_prefix + '{epoch:03d}' + weights_suffix
    checkpoint = ModelCheckpoint(weight_fname, monitor='val_loss', verbose=0, save_best_only=False,
                                 save_weights_only=True)
    # Train model
    train_history = autoencoder.fit(current_data, current_data, epochs=epoch_num, batch_size=batch_size, verbose=1,
                              shuffle=True, validation_data=(current_data, current_data), callbacks=[checkpoint],
                              initial_epoch=latest_idx+1)
    # Update history file with new training history
    epoch_history.append_history(train_history.history)
    history_list.append(epoch_history)
    if visualize_flag:
        print 'Visualizing the training and validation accuracies for stage %s'%n
        epoch_history.visualize_accuracy()
        plt.clf()
        current_epoch = int(input('Please specify which epoch to use for predicting (max %s)'%(epoch_num-1)))
        encoder_list[n].load_weights(weights_path + n_prefix + format_epoch(current_epoch) + weights_suffix,
                                     by_name=True)
        epoch_idx_params['param%s'%n] = current_epoch
    else:
        epoch_idx_params['param%s'%n] = get_current_epoch(n_prefix)
    # Transform current data with current encoder layer
    current_data = encoder_list[n].predict(current_data)
    print('Shape of current data: {} Rows, {} Columns'.format(*current_data.shape))

# Create transformations
current_data = scaled_data
for i, encoder in enumerate(encoder_list):
    print '\nApplying transformation on %s autoencoder stack'%i
    n_prefix = '%s_%s'%(str(n), weights_prefix)
    # Option to select specific epoch
    if visualize_flag:
        current_epoch = epoch_idx_params['param%s'%i]
    else:  # Set latest epoch as current epoch
        current_epoch = get_current_epoch(n_prefix)
    # Load old weights (if exists)
    prior_weight_fname = weights_path + n_prefix + format_epoch(current_epoch) + weights_suffix
    if os.path.exists(prior_weight_fname):
        print 'Loading existing model weight...\n', prior_weight_fname
        encoder.load_weights(prior_weight_fname, by_name=True)
        print 'Reducing dimensionality for autoencoder layer %s'%i
        current_data = encoder.predict(current_data)
        # Split train and test
        transformed_train = pd.DataFrame(data=current_data[:num_train, :], index=all_idx[:num_train])
        transformed_test = pd.DataFrame(data=current_data[num_train:, :], index=all_idx[num_train:])
        # Save train and test
        print 'Saving transformed train and test sets for AE layer %s'%i
        transformed_train.to_csv(trans_path + 'train_' + str(transformed_train.shape[1]) + trans_suffix)
        transformed_test.to_csv(trans_path + 'test_' + str(transformed_test.shape[1]) + trans_suffix)
    else:
        print 'No trained epochs exist for autoencoder layer %s'%i
        continue
