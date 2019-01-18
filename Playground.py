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
import time
import os
import pickle

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
number_files_dataset = 80
train_test_ratio = 0.75
use_all_addr = False

[dataset_train, dataset_test, labels_train, labels_test] = on_off_load(number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr)

#%% Network setting and feature exctraction (aka basis learning)

# Network settings
# =============================================================================
# feat_number: is the number of feature or centers used by the 0D network
# feat_size: is the length of the time context generated per each spikeoo
# taus: is a list containing the time coefficient used for the time surface creations
#       for each channel
# taucoef : Moltiplication factor for all taus
# =============================================================================


basis_number = [[10,10],[10,10]] 
context_lengths = [8,10]
input_channels = 32 + 32*use_all_addr

channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 
                                                             # cochlea used for this datasets
second_layer_taus = np.ones(basis_number[1][0]) # The taus for this layer are homogeneous across all channels
#third_layer_taus = np.ones(basis_number[2][0]) # The taus for this layer are homogeneous across all channels
taus_T_coeff = np.array([0.5,500]) # Multiplicative coefficients to help to change quickly the taus_T

taus_T = (taus_T_coeff*[channel_taus,second_layer_taus]).tolist()
taus_2D = [500,500]

# Create the network
Net = Solid_HOTS_Net(basis_number, context_lengths, input_channels, taus_T, taus_2D,
                     exploring=True, net_seed = 0)

# Learn the feature
Net.learn(dataset_train)
