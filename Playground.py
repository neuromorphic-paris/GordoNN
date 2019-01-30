# -*- coding: utf-8 -*-
"""
Created on Mar Jan 15 11:21:12 2019

@author: marcorax, pedro

This script serve as a test for different implementations, and parameter exploration 
of HOTS under the name gordoNN to be tested with multiple speech recognition tests.

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
import os
import pickle
import datetime

# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load

# 3D Dimensional HOTS or Solid HOTS
from Libs.Solid_HOTS.Solid_HOTS_Network import Solid_HOTS_Net

# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'


# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 12 # seed used for dataset shuffling if set to 0 the process will be totally random

#%% ON OFF Dataset
# Two class of recordings are used. The first class is composed by files containing
# a single word each, "ON", the second class is equal but the spelled word is "OFF"
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class (On, Off)
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channel of the cochlea
# =============================================================================

number_files_dataset = 60
train_test_ratio = 0.75
use_all_addr = False
number_of_labels = 2
parameter_folder = "Parameters/On_Off/"

legend = ("On","Off") # Legend containing the labes used for plots


[dataset_train, dataset_test, labels_train, labels_test,_,_] = on_off_load(number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr)


#%% Network setting and feature exctraction (aka basis learning)

# Network settings
# =============================================================================
# basis_number(list of int lists): the number of feature or centers used by the Solid network
#                             the first index identifies the layer, the second one
#                             is 0 for the centers of the 0D sublayer, and 1 for 
#                             the 2D centers
# context(list of int): the length of the time context generatef per each layer
# input_channels(int): the total number of channels of the cochlea in the input files 
# taus_T(list of float lists):  a list containing the time coefficient used for     
#                              the context creations for each layer (first index)
#                              and each channel (second index) 
# taus_2D(list of float):  a list containing the time coefficients used for the 
#                          creation of timesurfaces per each layer
# exploring(boolean) : If True, the network will output messages to inform the 
#                      the users about the current states and will save the 
#                      basis at each update to build evolution plots (currently not 
#                      available cos the learning is offline)
# net_seed : seed used for net generation, if set to 0 the process will be totally random
# =============================================================================


basis_number = [[8,40],[10,40]] 
context_lengths = [8,10,40]
input_channels = 32 + 32*use_all_addr

channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 
                                                             # cochlea used for this datasets
second_layer_taus = np.ones(basis_number[0][1]) # The taus for this layer are homogeneous across all channels
#third_layer_taus = np.ones(basis_number[1][1]) # The taus for this layer are homogeneous across all channels
taus_T_coeff = np.array([0.5,50000]) # Multiplicative coefficients to help to change quickly the taus_T

taus_T = (taus_T_coeff*[channel_taus,second_layer_taus]).tolist()
taus_2D = [3000,3000,500000]

# Create the network
Net = Solid_HOTS_Net(basis_number, context_lengths, input_channels, taus_T, taus_2D,
                     exploring=True, net_seed = 0)

# Learn the feature
Net.learn(dataset_train)

#%% Classification train

number_of_labels = len(legend)
Net.histogram_classification_train(labels_train,number_of_labels)

# Plotting results
Net.plot_histograms(legend)
plt.show() 
 
#%% Classification test 
prediction_rate, distances, predicted_labels = Net.histogram_classification_test(labels_test,number_of_labels,dataset_test)

# Plotting results
print("Euclidean distance recognition rate :             "+str(prediction_rate[0]))
print("Normalsed euclidean distance recognition rate :   "+str(prediction_rate[1]))
print("Bhattachaya distance recognition rate :           "+str(prediction_rate[2]))

Net.plot_histograms(legend, labels=labels_test)
plt.show()  

#%% Plot Basis 
#TODO add more information, time or channel and feature axes 
layer = 0
sublayer = 1
Net.plot_basis(layer, sublayer)
plt.show()    

#%% Save network parameters

now=datetime.datetime.now()
file_name = "GordoNN_Params_2L_8_"+str(now)+".pkl"
with open(parameter_folder+file_name, 'wb') as f:
    pickle.dump([basis_number, context_lengths, input_channels, taus_T, taus_2D], f)