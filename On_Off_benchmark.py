# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: marcorax, pedro

This file serve a benchmark for different implementations of HOTS under the name
of gordoNN to be tested with a simple binary classification test.
Two class of recordings are used. The first class is composed by files containing
a single word each ("ON"), the second class is equal but the spelled word is OFF

HOTS (The type of neural network implemented in this project) is a machine learning
method totally unsupervised.

To test if the algorithm is learning features from the dataset, a simple classification
task is accomplished with the use of histogram classification, taking the activity
of the last layer.

If by looking at the activities of the last layer we can sort out the class of the
input data, it means that the network has managed to separate the classes.

If the results are good enough, more tests more complicated than this might follow.
   
 
"""
# General Porpouse Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Data loading Libraries
from Libs.Data_loading.AERDATA_load import AERDATA_load
from Libs.Data_loading.get_filenames_dataset import get_filenames_on_off_dataset

# 0 dimensional HOTS
from Libs.HOTS_0D.HOTS_0D_Network import HOTS_0D_Net

# 2 dimensional HOTS
from Libs.HOTS_2D.HOTS_Sparse_Network import HOTS_Sparse_Net
from Libs.HOTS_2D.Time_Surface_generators import Time_Surface_all

# Dataset Parameters
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class (On, Off)
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# shuffle_seed : The seed used to shuffle the data, if 0 it will be totally 
#                random (no seed used)
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channel of the cochlea
# =============================================================================

number_files_dataset = 80
train_test_ratio = 0.75
shuffle_seed = 12
use_all_addr = False

addresess_number = 32 + 32*use_all_addr

## Data Loading

print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
start_time = time.time()
[filenames_train, labels_train, filenames_test, labels_test] = get_filenames_on_off_dataset(number_files_dataset, train_test_ratio, shuffle_seed)
print("Getting filenames from the dataset took %s seconds." % (time.time() - start_time))

## Reading spikes from each of the files
print ('\n--- READING SPIKES ---')
start_time = time.time()

dataset_train = []
dataset_test = []

for train_file in range(len(filenames_train)):
    addresses, timestamps = AERDATA_load(filenames_train[train_file], use_all_addr)
    dataset_train.append([np.array(timestamps), np.array(addresses)])
for test_file in range(len(filenames_test)):
    addresses, timestamps = AERDATA_load(filenames_test[test_file], use_all_addr)
    dataset_test.append([np.array(timestamps), np.array(addresses)])

# Shuffle data
# Setting the random state for data shuffling
rng = np.random.RandomState()
if(shuffle_seed!=0):
    rng.seed(shuffle_seed+1)

# Shuffle the dataset and the labels with the same order
combined_data = list(zip(dataset_train, labels_train))
rng.shuffle(combined_data)
dataset_train[:], labels_train[:] = zip(*combined_data)

combined_data = list(zip(dataset_test, labels_test))
rng.shuffle(combined_data)
dataset_test[:], labels_test[:] = zip(*combined_data)

print("Reading spikes took %s seconds." % (time.time() - start_time))



#%% 0D Network setting and feature exctraction (aka basis learning)

# 0D Network settings
# =============================================================================
# feat_number: is the number of feature or centers used by the 0D network
# feat_size: is the length of the time context generated per each spike
# taus: is a list containing the time coefficient used for the time surface creations
#       for each channel
# taucoef : Moltiplication factor for all taus
# =============================================================================

feat_number = 20 
feat_size = 8

taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
        1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
         20126, 25227, 31621, 39636, 49682])

taucoeff = 0.5 

# Create the network
Net_0D = HOTS_0D_Net(feat_number, feat_size, taucoeff*taus)

# Learn the feature
Net_0D.learn_offline(dataset_train)

# Update the dataset train with polarity (bunch of zeros) and 0D features
# to be computed by the second half of GordoNN
dataset_train = Net_0D.net_response

#%%  Histograms computation and test 

# Test result]
# =============================================================================
# total_net_result : the recognition rate of the entire network, computed using
#                    three different distances 1:euclidean 
#                    2:euclidean normalized on the number of spikes                   
#                    3: bhattacharyya distance between normalized histograms
# 
# total_net_result : same as total_net_result but per each address
# =============================================================================

# The total number of labels for the given problem
number_of_labels = 2
# Computing the two signatures for the entire network and for each address
Net_0D.histogram_classification_train(labels_train, number_of_labels, addresess_number)
# Computing the rate per each channel and for the entire network
total_net_result, channeled_results = Net_0D.histogram_classification_test(labels_test, number_of_labels, addresess_number,dataset_test)

# Update the dataset test with polarity (bunch of zeros) and 0D features
# to be computed by the second half of GordoNN
dataset_test = Net_0D.net_response

#%% Generate 2D net

# Plotting settings
plt.style.use("dark_background")

# 2D Network settings
# =============================================================================
# basis_number is a list containing the number of basis used for each layer
# basis_dimension: is a list containing the dimension of every base for each layer
# taus: is a list containing the time coefficient used for the time surface creations
#       for each layer, all three lists need to share the same lenght obv.
# first_layer_polarities : the number of distinct polarities in the input layer 
# shuffle_seed, net_seed : seed used for dataset shuffling and net generation,
#                       if set to 0 the process will be totally random
# delay_coeff : the coefficient used to linearly adress delay to outputs of each layer
#               out_timestamp = in_timestamp + delay_coeff*(1-abs(a_j))
# =============================================================================

basis_number = [10]
basis_dimension = [[Net_0D.feat_number, addresess_number]] 
taus = [10000]
# The output of the first layer Hots is monopolar
first_layer_polarities = 1
shuffle_seed = 7
net_seed = 25

delay_coeff = 15000    
    
# Print a series of elements to check if it's all right
#file = 3
#
#tsurface=Time_Surface_all(xdim=basis_dimension[0][0], ydim=basis_dimension[0][1], timestamp=dataset_train[file][0][10], timecoeff=taus[0], dataset=dataset_train[file], num_polarities=1, minv=0.1, verbose=False)
#ax = sns.heatmap(tsurface, annot=False, cbar=False, vmin=0, vmax=1)
#plt.show()
#
#for i in range(10030,10170):
#    print(i)
#    tsurface=Time_Surface_all(xdim=basis_dimension[0][0], ydim=basis_dimension[0][1], timestamp=dataset_train[file][0][i], timecoeff=taus[0], dataset=dataset_train[file], num_polarities=1, minv=0.1, verbose=False)
#    sns.heatmap(data=tsurface, ax=ax, annot=False, cbar=False, vmin=0, vmax=1)
#    plt.draw()
#    plt.pause(0.001)

# Generate the network
Net = HOTS_Sparse_Net(basis_number, basis_dimension, taus, first_layer_polarities, delay_coeff, net_seed)
    
#%% Learning-online-Exp distance and Thresh

print ('\n--- 2D HOTS feature extraction ---')
start_time = time.time()

sparsity_coeff = [1, 1, 2000000]
learning_rate = [1, 1, 6000]
noise_ratio = [1, 0, 500]
sensitivity = [0.01, 0.01, 400000]
channel = 9

Net.learn_online(dataset=dataset_train,
                 channel = channel,
                 method="Exp distance", base_norm="Thresh",
                 noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                 sensitivity=sensitivity,
                 learning_rate=learning_rate, verbose=False)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

# Taking the steady state values to perform the other tests
sparsity_coeff = sparsity_coeff[1]
noise_ratio = noise_ratio[1]
sensitivity = sensitivity[1]

#%% Learning offline full batch

#start_time = time.time()
#
#sparsity_coeff = 0.8
#learning_rate = 0.2        
#max_steps = 5
#base_norm_coeff = 0.0005
#precision = 0.01
#channel = 5
#
#Net.learn_offline(dataset_train, channel, sparsity_coeff, learning_rate, max_steps, base_norm_coeff, precision, verbose=False)
#    
#elapsed_time = time.time()-start_time
#print("Learning elapsed time : "+str(elapsed_time))           
#sensitivity = 0   
#noise_ratio = 0 
#%% Plot Basis 

#layer = 0
#sublayer = 0
#Net.plot_basis(layer, sublayer)
#plt.show()    

#%% Classification train

#net_activity = Net.full_net_dataset_response(dataset_testing, channel, "Exp distance", 
#                                                      noise_ratio, 
#                                                      sparsity_coeff,
#                                                      sensitivity)

Net.histogram_classification_train(dataset_train, channel,
                                   labels_train, 
                                   2, "Exp distance", noise_ratio,
                                   0, sensitivity)



#%% Classification test 

test_results = Net.histogram_classification_test(dataset_test, channel,
                                                 labels_test,
                                                 2, "Exp distance", noise_ratio,
                                                 0, sensitivity) 
hist = np.transpose(Net.histograms)
norm_hist = np.transpose(Net.normalized_histograms)
test_hist = np.transpose(test_results[2])
test_norm_hist = np.transpose(test_results[3])

eucl = 0
norm_eucl = 0
bhatta = 0
for i,right_label in enumerate(labels_test):
    eucl += (test_results[1][i][0] == right_label)/len(labels_test)
    norm_eucl += (test_results[1][i][1] == right_label)/len(labels_test)
    bhatta += (test_results[1][i][2] == right_label)/len(labels_test)
