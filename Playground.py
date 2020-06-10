# -*- coding: utf-8 -*-
"""
Created on Mar Jan 15 11:21:12 2019

@author: marcorax, pedro

This script serve as a test for different implementations, and parameter exploration 
of HOTS under the name gordoNN to be tested with multiple speech recognition tests.

HOTS (The type of neural network implemented in this project) is a machine learning
method totally unsupervised.


To test if the algorithm is learning features from the dataset, a simple classification
task is accomplished with the use multiple classifiers, taking the activity
of the last layer.

If by looking at the activities of the last layer we can sort out the class of the  
input data, it means that the network has managed to separate the classes.
   
"""

# General Porpouse Libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# to use CPU for training
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load

# 3D Dimensional HOTS or Solid HOTS
from Libs.Solid_HOTS.Network import Solid_HOTS_Net
from Libs.Solid_HOTS._General_Func import first_layer_sampling_plot, other_layers_sampling_plot, first_layer_sampling, recording_local_surface_generator

# To avoid MKL inefficient multithreading
os.environ['MKL_NUM_THREADS'] = '1'

# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 25 # seed used for dataset shuffling if set to 0 the process will be totally random 

#%% ON OFF Dataset
# Two class of recordings are used. The first class is composed by recordings
# of the word: "OFF", the second class is composed by s"ON"
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class (On, Off)
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channels of the cochlea
# =============================================================================

number_files_dataset = 600
train_test_ratio = 0.75
use_all_addr = False
number_of_labels = 2
label_file = "Data/On_Off/files_timestamps.csv"
classes = ("Off","On") 



## IF YOU WANT TO LOAD A PRECISE SET OF FILENAMES IN CASE THE SEED IS NOT RELIABLE AMONG DIFFERENT COMPUTERS
#
#filenames_train=np.load("filenames_train.npy")
#filenames_test=np.load("filenames_test.npy")
#labels_train=np.load("labels_train.npy")
#labels_test=np.load("labels_test.npy")
#
#[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, wordpos_train, wordpos_test] = on_off_load(number_files_dataset, label_file, train_test_ratio, 
#                                                                                             shuffle_seed, use_all_addr, filenames_train, filenames_test,
#                                                                                             labels_train, labels_test)



[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, wordpos_train, wordpos_test] = on_off_load(number_files_dataset, label_file, train_test_ratio, shuffle_seed, use_all_addr)


## IF YOU WANT TO SAVE A PRECISE SET OF FILENAMES IN CASE THE SEED IS NOT RELIABLE AMONG DIFFERENT COMPUTERS
#np.save("filenames_train",filenames_train)
#np.save("filenames_test",filenames_test)
#np.save("labels_train",labels_train)
#np.save("labels_test",labels_test)gc.collect()


#%% Network settings
# =============================================================================
#   features_number (nested lists of int) : the number of feature or centers used by the Solid network,
#                               the first index identifies the layer, the second one
#                               is the number of  for units of the 0D sublayer,
#                               and the third for the 2D units
#   l1_norm_coeff (nested lists of int) : Same structure of feature_number but used to store l1 normalization to sparify 
#                                         bottleneck layer of each autoencoder
#   learning_rate (nested lists of int) : Same structure of feature_number but 
#                                         used to store the learning rates of 
#                                         each autoencoder
#   epochs (nested lists of int) : Same structure of feature_number but 
#                                         used to store the epochs to train 
#                                         each autoencoder
#   local_surface_lengths (list of int): the length of the Local time surfaces generated per each layer
#   input_channels (int) : thex total number of channels of the cochlea in the input files 
#   taus_T(list of float lists) :  a list containing the time coefficient used for 
#                                  the local_surface creations for each layer (first index)
#                                  and each channel (second index). To keep it simple, 
#                                  it's the result of a multiplication between a vector for each 
#                                  layer(channel_taus) and a coefficient (taus_T_coeff).
#   taus_2D (list of float) : a list containing the time coefficients used for the 
#                            creation of timesurfaces per each layer
#   batch_size (list of int) : a list containing the batch sizes used for the 
#                            training of each layer
#   activity_th (float) : The code will check that the sum(local surface)150
#   intermediate_dim_T (int) : Number of units used for intermediate layers
#   intermediate_dim_2D (int) : Number of units used for intermediate layers
#   threads (int) : The network can compute timesurfaces in a parallel way,
#                   this parameter set the number of multiple threads allowed to run
#
#
#   exploring (boolean) : If True, the network will output messages to inform the 
#                         the users about the current states and will save the 
#                         basis at each update to build evolution plots (currently not 
#                         available cos the learning is offline)
# =============================================================================

"""
features_number=[[6,10],[15,20]] 
l1_norm_coeff=[[0,0],[0,0]]
learning_rate = [[3e-4,3e-4],[1e-4,1e-4]]
epochs = [[900,900],[900,900]]
local_surface_lengths = [20,500]
input_channels = 32 + 32*use_all_addr
"""
features_number=[[6,10]] 
l1_norm_coeff=[[0,0]]
learning_rate = [[3e-4,3e-4]]
epochs = [[10,10]]         #epochs = [[1,1]] #epochs = [[900,900]]
local_surface_lengths = [20]
input_channels = 32 + 32*use_all_addr

### Channel Taus ###

##Theorical _Probably Wrong
#channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
#                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
#                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 

## Uniform
#channel_taus = np.arange(32)*1

#Linear interpolation between highest spike frequency 90ks/s to lowest 20ks/s, used to balance the filters
channel_taus = np.arange(1,9,(9-1)/32)
                                                             
taus_T_coeff = np.array([15000,500000]) # Multiplicative coefficients to help to change quickly the taus_T
taus_T = (taus_T_coeff*[channel_taus,np.ones(features_number[0][1])]).tolist()
spacing_local_T = [1,2]
taus_2D = [50000,500000]  

batch_size = [512,512] #batch_size = [200000,200000]

activity_th = 0
intermediate_dim_T=20
intermediate_dim_2D=90

threads=1 # Due to weird problem for memory access the data is copied,
          # if you don't have enough ram, decrease this or the number of files 
          # or layers
          
exploring=True

network_parameters = [[features_number, l1_norm_coeff, learning_rate, 
                       local_surface_lengths, input_channels, taus_T, taus_2D, 
                 threads, exploring],[learning_rate, epochs, l1_norm_coeff,
                 intermediate_dim_T, intermediate_dim_2D, activity_th, 
                 batch_size, spacing_local_T]]


#%% Network creation and learning 

# Create the network
Net = Solid_HOTS_Net(network_parameters)


# Learn the features
Net.learn(dataset_train, dataset_test)


#%% Methods to add layers/rerun training

# # If you want to add a layer or more on top of the net and keep the results
# # you had for the first ones use this. 
# # You will have to load new parameters and run the learning from the 
# # index of the new layer (You computed a 2 layer network, you want to add 2 more,
# # you use a new set of parameters, and rerun learning with rerun_layer=2, as the 
# # layers index start with 0.
# Net.add_layers(network_parameters)
# Net.learn(dataset_train, dataset_test, rerun_layer = 2)

# # If you want to recompute few layers of the net in a sequential manner
# # change and load the parameters with this method and then rerun the net learning
# # (For example you computed a 2 layer network, and you want to rerun the second layer,
# # you use a new set of parameters, and rerun learning with rerun_layer=1, as the 
# # layers index start with 0.
# Net.load_parameters(network_parameters)
# Net.learn(dataset_train, dataset_test, rerun_layer = 1)


#%% LSTM classifier training
# Simple LSTM applied on all output events, binned in last_bin_width-size windows.

lstm_bin_width = 150
lstm_sliding_amount = 10
lstm_units = 50
lstm_learning_rate = 1e-4
lstm_epochs = 5
lstm_batch_size = 64
lstm_patience = 30

Net.lstm_classification_train(labels_train, labels_test, number_of_labels, lstm_bin_width, 
                              lstm_sliding_amount, lstm_learning_rate, lstm_units, lstm_epochs, 
                              lstm_batch_size, lstm_patience)
gc.collect()

#%% LSTM test
threshold = 0.6
Net.lstm_classification_test(labels_test, number_of_labels, lstm_bin_width, 
                             lstm_sliding_amount, lstm_batch_size, threshold )

#%%
tmp=Net.last_layer_activity_test
i=0
for recording in range(len(tmp)):
    i+=len(tmp[recording][0])
    
 
#%% Mlp classifier training
# Simple MLP applied on all output events as a weak classifier to prove HOTS
# working.

mlp_learning_rate = 1e-4
mlp_epochs = 15000
mlp_hidden_size = 10
mlp_batch_size = 200000
patience = 500

Net.mlp_classification_train(labels_train, labels_test, number_of_labels, mlp_learning_rate,
                             mlp_hidden_size, mlp_epochs, mlp_batch_size, patience)
gc.collect()

#%% Mlp classifier testing
threshold = 0.8
Net.mlp_classification_test(labels_test, number_of_labels, mlp_batch_size,
                            threshold)

#%% Histogram mlp classifier training
# Simple MLP applied over the histogram (the summed response of the last layer
# for each recording) of the net activity.

hist_mlp_learning_rate = 4e-4
hist_mlp_epochs = 5000
hist_mlp_hidden_size = 5
hist_mlp_batch_size = 180
patience = 50

Net.hist_mlp_classification_train(labels_train, labels_test, number_of_labels, 
                                  hist_mlp_learning_rate, hist_mlp_hidden_size, 
                                  hist_mlp_epochs, hist_mlp_batch_size, patience)
gc.collect()

#%% Mlp hist classifier testing
threshold=0.5
Net.hist_mlp_classification_test(labels_test, number_of_labels,
                                 hist_mlp_batch_size, threshold)

#%% Print Surfaces
# Method to plot reconstructed and original surfaces
Net.plt_surfaces_vs_reconstructions(file=0, layer=0, test=False)

#%% Net history plot
# Method to print loss history of the network
Net.plt_loss_history(layer=0)

#%% Print last layer activity 
# Method to plot last layer activation of the network
Net.plt_last_layer_activation(file=1, labels=labels_train, labels_test=labels_test,
                              classes=classes, test=True)

     
#%% Reverse activation
# Method to plot reverse activation of a sublayer output (output related to input)
Net.plt_reverse_activation(file=0, layer=1, sublayer=0, labels=labels_train, 
                           labels_test=labels_test, classes=classes, test=True)



