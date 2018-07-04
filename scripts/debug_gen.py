# Generates debugging files and stores in data directory
import pandas as pd


# Load data
data_path = '../data/'
train_path = data_path + 'train.csv'
test_path = data_path + 'test.csv'
print 'Loading train and test datasets...'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Randomly sample from train and test and save debugging files
print 'Randomly sampling from train and test datasets...'
train_debug = train_df.sample(n=100, random_state=0)
test_debug = test_df.sample(n=200, random_state=0)

# Save debugging datasets
print 'Saving debugging datasets...'
train_debug.to_csv(data_path+'train_debug.csv')
test_debug.to_csv(data_path+'test_debug.csv')
