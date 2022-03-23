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
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import seaborn as sns
import os, gc, pickle
from tensorflow.keras.callbacks import EarlyStopping
from Libs.Solid_HOTS._General_Func import create_mlp


# To use CPU for training
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# To allow more memory to be allocated on gpus incrementally
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load, data_load

# 3D Dimensional HOTS or Solid HOTS
from Libs.Solid_HOTS.Network import Solid_HOTS_Net
# from Libs.Solid_HOTS._General_Func import first_layer_sampling_plot, other_layers_sampling_plot, first_layer_sampling, recording_local_surface_generator

# To avoid MKL inefficient multithreading (helped for some AMD processors before)
# now disabled
# os.environ['MKL_NUM_THREADS'] = '1'

# Plotting settings
sns.set(style="white")
# plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 25 # seed used for dataset shuffling if set to 0 the process will be totally random 
# shuffle_seed = 0 



#%% Google Commands Dataset small
# 30 class of recordings are used. They consists of simple keywords
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class 
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channels of the cochlea
# =============================================================================


number_files_dataset = 1000

train_test_ratio = 0.80


use_all_addr = False
number_of_labels = 8

dataset_folder ='Data/Google_Command'
spacing=1
# classes=['off','on']
classes=['stop', 'left', 'no', 'go', 'yes', 'down', 'right', 'up']

[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, classes] = data_load(dataset_folder, number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr, spacing, class_names=classes)


#%% Network settings
# =============================================================================
#   features_number (nested lists of int) : the number of feature or centers used by the Solid network,
#                               the first index identifies the layer, the second one
#                               is the number of  for units of the 0D sublayer,
#                               and the third for the 2D units
#   local_surface_lengths (list of int): the length of the Local time surfaces generated per each layer
#   input_channels (int) : thex total number of channels of the cochlea in the input files 
#   taus_T(list of float lists) :  a list containing the time coefficient used for 
#                                  the local_surface creations for each layer (first index)
#                                  and each channel (second index). To keep it simple, 
#                                  it's the result of a multiplication between a vector for each 
#                                  layer(channel_taus) and a coefficient (taus_T_coeff).
#   taus_2D (list of float) : a list containing the time coefficients used for the 
#                            creation of timesurfaces per each layer
#   activity_th (float) : The code will check that the sum(local surface)
#   threads (int) : The network can compute timesurfaces in a parallel way,
#                   this parameter set the number of multiple threads allowed to run
#
#
#   verbose (boolean) : If True, the network will output messages to inform the 
#                         the users about the current states and will save the 
#                         basis at each update to build evolution plots (currently not 
#                         available cos the learning is offline)
# =============================================================================


features_number=[[20,256]] 

# local_surface_lengths = [10,1]
local_surface_lengths = [50,1]
cross_surface_width = [-1,1]


input_channels = 32 + 32*use_all_addr

### Channel Taus ###

#Linear interpolation between highest spike frequency 90ks/s to lowest 20ks/s, used to balance the filters
# channel_taus = np.linspace(2,9,32)

#NEW LOGISTIC INTERPOLATION         
channel_taus=1/(np.array([2.        , 2.09818492, 2.19739597, 2.30845776, 2.42444764,
        2.54123647, 2.67160239, 2.80743423, 2.94395277, 3.09176148,
        3.24720257, 3.41043852, 3.57786533, 3.75780207, 3.94362851,
        4.14023903, 4.34601214, 4.56200522, 4.78905638, 5.02749048,
        5.27770753, 5.53978449, 5.81554546, 6.10468077, 6.40818066,
        6.72679013, 7.06129558, 7.41233318, 7.78086536, 8.16764435,
        8.57370814, 9.        ])[::-1]) 

channel_taus = (channel_taus/channel_taus[0])*2
                                                                                               

taus_2D = [100000]  

# n_batch_files = 2048
n_batch_files = 512

#dataset_runs = 10 #how many times the dataset is run for clustering
dataset_runs = 20 #how many times the dataset is run for clustering


threads=24 
          
verbose=True




#%% First layer decay search

# Tau_T_first = np.arange(200,2200,200)
Tau_T_first = np.arange(200,20200,200)

eucl_res= []
euclnorm_res = []

for Tau_T in Tau_T_first:
    
    taus_T_coeff = np.array([Tau_T,1]) # Multiplicative coefficients to help to change quickly the taus_T  #1000
    taus_T = (taus_T_coeff*[channel_taus,np.ones(256)]).tolist()
    
    network_parameters = [[features_number, local_surface_lengths, cross_surface_width, input_channels, taus_T, taus_2D, 
                 threads, verbose],[n_batch_files, dataset_runs]]
    
    # Create the network
    Net = Solid_HOTS_Net(network_parameters)
    
        
    # Learn the features
    Net.learn(dataset_train)
    Net.infer(dataset_test)


    eucl_res_tau, euclnorm_res_tau = Net.hist_classification_test(labels_train,labels_test,number_of_labels)
    eucl_res.append(eucl_res_tau)
    euclnorm_res.append(euclnorm_res_tau)


#%% Save Layer results
layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res, 'Taus_T' : Tau_T_first}

with open('Results/Decay_search/Layer_1_512_batch_20runs_20Features_newatt_rule_longer_mult_longer_sweep.pickle', 'wb') as handle:
    pickle.dump(layer_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# #%% Load Layer results

# with open('Results/Decay_search/Layer_1.pickle', 'rb') as handle:
#     layer_res = pickle.load(handle)
# 

#%% Load Layer results

with open('Results/Decay_search/Layer_1_512_batch_20runs_20Features.pickle', 'rb') as handle:
    layer_res_old = pickle.load(handle)

#%% Comparison HOTS layer and Hack Layer
plt.figure()
plt.plot(layer_res_old['Taus_T'], layer_res_old['Norm_eucl_res'], label="Att Original Approximation")
plt.plot(layer_res['Taus_T'], layer_res['Norm_eucl_res'], label="Att New Approximation")
plt.xlabel("Tau first layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Parameter search for 2 layers Audio HOTS")
plt.legend()

#%% Distribution of 

