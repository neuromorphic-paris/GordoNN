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
import os
import shutil
import csv
import pickle
import datetime
import gc
# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load
# Benchmark libs
from Libs.Benchmark_Libs import bench
# 3D Dimensional HOTS or Solid HOTS
from Libs.Solid_HOTS.Solid_HOTS_Network import Solid_HOTS_Net

## to use CPU for training
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# To avoid MKL inefficient multythreading
os.environ['MKL_NUM_THREADS'] = '1'


# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

result_folder = "Results/On_Off/Test_1L_old_data_new_bigger_net"

if os.path.exists(result_folder) : 
    print("Use a different folder")
#shuffle_seed = [315, 772, 164, 787, 446, 36, 234, 196, 994, 304, 637, 15, 206, 575, 846, 272, 443, 209, 653, 838] # seed used for dataset shuffling if set to 0 the process will be totally random 
#shuffle_seed = [24, 13, 26]

shuffle_seed = [24, 24, 24, 25, 25, 25, 26, 27, 28, 29, 315, 772, 164, 787, 446, 36, 234, 196, 994, 304, 637, 15, 206]
nthreads=8

#%% Selecting the dataset

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

number_files_dataset = 120
train_test_ratio = 0.75
use_all_addr = False
number_of_labels = 2
label_file = "Data/On_Off/files_timestamps.csv"

legend = ("On","Off") # Legend containing the labes used for plots




#%% Network setting and feature exctraction 

# Network settings
# =============================================================================
#   features_number (nested lists of int) : the number of feature or centers used by the Solid network,
#                               the first index identifies the layer, the second one
#                               is 0 for the centers of the 0D sublayer, and 1 for 
#                               the 2D centers
#   context_lengths (list of int): the length of the time context generatef per each layer
#   input_channels (int) : thex total number of channels of the cochlea in the input files 
#   taus_T(list of float lists) :  a list containing the time coefficient used for 
#                                  the context creations for each layer (first index)
#                                  and each channel (second index) 
#   taus_2D (list of float) : a list containing the time coefficients used for the 
#                            creation of timesurfaces per each layer
#   threads (int) : The network can compute timesurfaces in a parallel way,
#                   thi200s parameter set the number of multiple threads allowed to run
#   exploring (boolean) : If True, the network will output messages to inform the 
#                         the users about the current states and will save the 
#                         basis at each update to build evolution plots (currently not 
#                         available cos the learning is offline)
# =============================================================================



features_number=[[2,8,1]] #(These setting are generating too many events, i have to fix it )
context_lengths = [100,300,50,40,30]
input_channels = 32 + 32*use_all_addr
##### I HAD to HIGHER thE SPARSITY BECAUSE thE ACTivity was compressed between 8,6
l1_norm_coeff=[[0,0],[0,0],[1e-6,1e-6],[1e-6,1e-6],[1e-5,1e-5]]
#COMPUTED
#channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
#                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
#                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 
#                      
#MEAN MIN                       
#channel_taus = np.array([  6.04166667,   4.025     ,   2.66666667,   1.85      ,
#         1.98333333,   1.66666667,   1.40833333,   1.25      ,
#         1.86666667,   1.85      ,   1.78333333,   1.675     ,
#         1.55833333,   1.75      ,   1.81666667,   3.00833333,
#         3.03333333,   3.65      ,   4.8       ,   5.81666667,
#         7.49166667,  12.6       ,  15.35833333,  31.43333333,
#        24.01666667,  30.88333333,  51.9       ,  86.25833333,
#       123.56666667, 189.175     , 213.80833333, 229.08333333])

#channel_taus = np.array([  4174.,  14911.,   1790.,   5074.,   1905.,  16363.,   5244.,
#         7163.,   4732.,   7549.,   6514.,   2151.,   3733.,   3611.,
#         4794.,   9597.,   7913.,   5572.,   7948.,   9884.,   9464.,
#        18683.,  14012.,  22862.,  44167.,  35645.,  45560., 141597.,
#        64530.,  94234., 110459., 132337.])

channel_taus = np.ones(32)*4
                                                             
second_layer_taus = np.ones(features_number[0][1]) # The taus for this layer are homogeneous across all channels
#third_layer_taus = np.ones(features_number[1][1]) # The taus for this layer are homogeneous across all channels
#fourth_layer_taus = np.ones(features_number[2][1]) # The taus for this layer are homogeneous across all channels
#fifth_layer_taus = np.ones(features_number[3][1]) # The taus for this layer are homogeneous across all channels[4,8]
third_layer_taus = np.ones(0) # The taus for this layer are homogeneous across all channels
fourth_layer_taus = np.ones(0) # The taus for this layer are homogeneous across all channels
fifth_layer_taus = np.ones(0) # The taus for this layer are homogeneous across all channels
taus_T_coeff = np.array([5000, 500000, 500000, 500000, 800000]) # Multiplicative coefficients to help to change quickly the taus_T

taus_T = (taus_T_coeff*[channel_taus, second_layer_taus, third_layer_taus, fourth_layer_taus, fifth_layer_taus]).tolist()
taus_2D = [100000, 500000, 500000, 50000, 800000]  



learning_rate = [[5e-4,5e-4],[5e-4,5e-4],[1e-3,1e-3],[5e-4,5e-4],[5e-4,5e-4]]
epochs = [[15,40],[80,80],[20,20],[300,300]]

cross_correlation_th_array=[0, 0, 0, 0.1, 0.3]
batch_size = [4096]

spacing = [5,100,40]

exploring=False

# Mlp classifier settings
last=-0
number_of_labels=len(legend)
mlp_learning_rate = 1e-4                  
mlp_epochs=60
threshold=0.5

network_parameters = [[features_number, context_lengths, input_channels, taus_T, taus_2D, 
                 nthreads, exploring],[learning_rate, epochs, l1_norm_coeff,
                 intermediate_dim_T, intermediate_dim_2D, cross_correlation_th_array,\
                 batch_size, spacing]]

classifier_parameters = [[last, number_of_labels, mlp_epochs, mlp_learning_rate],[threshold]]

#%% Run the bench
results=[]
data_reference = []
for seed in shuffle_seed:
    print(seed)
    dataset_parameters = [number_files_dataset, label_file, train_test_ratio, seed, use_all_addr]
    single_run_results, filenames = bench(dataset_parameters, network_parameters, classifier_parameters)
    results.append(single_run_results)
    data_reference.append(filenames)
    
#%% Save Results
os.mkdir(result_folder)

with open(result_folder + "/results.csv", 'w', newline='') as csvfile:
    fieldnames = ['Seed', 'Prediction_rate']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(results)):
        writer.writerow({'Seed' : shuffle_seed[i], 'Prediction_rate' : results[i]})
                         
# filenames saving        
with open(result_folder + "/filenames.txt", "wb") as fp:   #Pickling
    pickle.dump(data_reference, fp)
     
# Save Parameters
    network_parameters = [[features_number, context_lengths, input_channels, taus_T, taus_2D, 
                 nthreads, exploring],[learning_rate, epochs, l1_norm_coeff,
                                    cross_correlation_th_array, batch_size, spacing]]

    classifier_parameters = [[last, number_of_labels, mlp_epochs, mlp_learning_rate],[threshold]]
    
parameter_dict = { "features_number" : features_number, "context_lengths" : context_lengths,
                  "input_channels" : input_channels, "taus_T" : taus_T, "taus_2D" : taus_2D, 
                 "nthreads" : nthreads, "exploring" : exploring, "learning_rate" : learning_rate,
                 "epochs" : epochs, "l1_norm_coeff" : l1_norm_coeff, "cross_correlation_th_array" : cross_correlation_th_array,
                 "batch_size" : batch_size, "spacing" : spacing, "last" : last,
                 "number_of_labels" : number_of_labels, "mlp_epochs" : mlp_epochs,
                 "mlp_learning_rate" : mlp_learning_rate, "threshold" : threshold }

with open(result_folder + "/parameters.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for key, value in parameter_dict.items():
        writer.writerow([key, value])