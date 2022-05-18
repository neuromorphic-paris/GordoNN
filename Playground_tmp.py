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
from joblib import Parallel, delayed 
from sklearn import svm
import pandas as pd


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

from Libs.GORDONN.Layers.Local_Layer import Local_Layer
from Libs.GORDONN.Layers.Cross_Layer import Cross_Layer
from Libs.GORDONN.Layers.Pool_Layer import Pool_Layer
from Libs.GORDONN.Network import GORDONN

# Plotting settings
sns.set(style="white")
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 25 # seed used for dataset shuffling if set to 0 the process will be totally random 
# shuffle_seed = 0 


#%% Google Commands Dataset
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
# number_files_dataset = 10

train_test_ratio = 0.80


use_all_addr = False
number_of_labels = 8

dataset_folder ='Data/Google_Command'
spacing=1
# classes=['off','on']
classes=['stop', 'left', 'no', 'go', 'yes', 'down', 'right', 'up']

[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, classes] = data_load(dataset_folder, number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr, spacing, class_names=classes)


#%% Network local layer

input_channels = 32 + 32*use_all_addr

channel_taus=1/np.array([2.        , 2.09818492, 2.19739597, 2.30845776, 2.42444764,
       2.54123647, 2.67160239, 2.80743423, 2.94395277, 3.09176148,
       3.24720257, 3.41043852, 3.57786533, 3.75780207, 3.94362851,
       4.14023903, 4.34601214, 4.56200522, 4.78905638, 5.02749048,
       5.27770753, 5.53978449, 5.81554546, 6.10468077, 6.40818066,
       6.72679013, 7.06129558, 7.41233318, 7.78086536, 8.16764435,
       8.57370814, 9.        ][::-1])   

channel_taus = channel_taus/channel_taus[0]

tau_K = 10000

taus = tau_K*channel_taus

# Create the network
local = Local_Layer(n_features=20, local_tv_length=20, 
                    n_input_channels=input_channels, taus=taus, 
                    n_batch_files=None, dataset_runs=1, n_threads=22,
                    verbose=True)

    
# Learn the features
local.learn(dataset_train)
tmp = local.features

# Predict the features
local_response = local.predict(dataset_test)
hists, norm_hist = local.gen_histograms(local_response)

sign, norm_sign= local.gen_signatures(hists, norm_hist, classes, labels_test)

file_i = 0
local.response_plot(local_response, f_index=file_i, class_name=classes[labels_test[file_i]])


#%% Network local layer

input_channels = 32 + 32*use_all_addr

df = pd.read_csv (r'whitenoise.csv')
mean_rate = np.asarray(df.columns[:], dtype=float)
channel_taus = 1/mean_rate

channel_taus = channel_taus/channel_taus[0]

tau_K = 5000

taus = tau_K*channel_taus

# Create the network
local = Local_Layer(n_features=20, local_tv_length=10, 
                    n_input_channels=input_channels, taus=taus, 
                    n_batch_files=None, dataset_runs=1, n_threads=22,
                    verbose=True)

    
# Learn the features
local.learn(dataset_train)
tmp = local.features

# Predict the features
local_response = local.predict(dataset_test)
hists, norm_hist = local.gen_histograms(local_response)

sign, norm_sign= local.gen_signatures(hists, norm_hist, classes, labels_test)

file_i = 0
local.response_plot(local_response, f_index=file_i, class_name=classes[labels_test[file_i]])
local.features_plot()

#%% Network cross layer stack 

# Create the network
cross = Cross_Layer(n_features=20, cross_tv_width=32, 
                    n_input_channels=input_channels, taus=50e3, 
                    n_input_features=20, n_batch_files=None,
                    dataset_runs=1, n_threads=22,
                    verbose=True)

    
# Learn the features
cross.learn(local_response)
tmp = cross.features


# Predict the features
cross_response = cross.predict(local_response)
hists, norm_hist = cross.gen_histograms(cross_response)

sign, norm_sign= cross.gen_signatures(hists, norm_hist, classes, labels_test)

file_i = 0
local.response_plot(local_response, f_index=file_i, class_name=classes[labels_test[file_i]])



#%% Network cross layer as first
 
input_channels = 32 + 32*use_all_addr

channel_taus=1/np.array([2.        , 2.09818492, 2.19739597, 2.30845776, 2.42444764,
       2.54123647, 2.67160239, 2.80743423, 2.94395277, 3.09176148,
       3.24720257, 3.41043852, 3.57786533, 3.75780207, 3.94362851,
       4.14023903, 4.34601214, 4.56200522, 4.78905638, 5.02749048,
       5.27770753, 5.53978449, 5.81554546, 6.10468077, 6.40818066,
       6.72679013, 7.06129558, 7.41233318, 7.78086536, 8.16764435,
       8.57370814, 9.        ][::-1])   

channel_taus = channel_taus/channel_taus[0]

tau_K = 10000

taus = tau_K*channel_taus

# Create the network
cross = Cross_Layer(n_features=20, cross_tv_width=3, 
                    n_input_channels=input_channels, taus=taus, 
                    n_input_features=1, n_batch_files=None,
                    dataset_runs=1, n_threads=22,
                    verbose=True)

# Learn the features
cross.learn(dataset_train)
tmp = cross.features

cross.predict(dataset_train)

# Predict the features
cross_response = cross.predict(dataset_train)

cross.learn(cross_response)


#%% Pool layer test


pool = Pool_Layer(n_input_channels=32, pool_factor=4)

tmp=pool.pool(dataset_train)
tmp=pool.pool(local_response)

#%% Network test



#First layer parameters
input_channels = 32 + 32*use_all_addr

df = pd.read_csv (r'whitenoise.csv')
mean_rate = np.asarray(df.columns[:], dtype=float)
channel_taus = 1/mean_rate

channel_taus = channel_taus/channel_taus[0]

tau_K = 500

taus = tau_K*channel_taus

n_features=20
local_tv_length=10
n_input_channels=input_channels
n_batch_files=None
dataset_runs=1

local_layer_parameters = [n_features, local_tv_length, n_input_channels, taus,\
                    n_batch_files, dataset_runs]
    

#Second layer parameters
n_input_features=n_features 
n_input_channels=input_channels
n_features=64
cross_tv_width=6 

taus=50e3
n_batch_files=None
dataset_runs=1


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   
             
Net = GORDONN(n_threads=24, verbose=True)
Net.add_layer("Local", local_layer_parameters)
Net.add_layer("Cross", cross_layer_parameters)
Net.learn(dataset_train,labels_train,classes)
Net.predict(dataset_test, labels_train, labels_test, classes)

print("Histogram accuracy: "+str(Net.layers[0].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[0].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[0].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[0].svm_norm_hist_accuracy))


print("Histogram accuracy: "+str(Net.layers[1].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[1].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[1].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[1].svm_norm_hist_accuracy))

#%% Parameter change and overload
   

#Second layer parameters
new_taus = 5e3
Net.layers[1].taus=new_taus

Net.learn(dataset_train,labels_train,classes, rerun_layer=1)
Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=1)

print("Histogram accuracy: "+str(Net.layers[0].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[0].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[0].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[0].svm_norm_hist_accuracy))


print("Histogram accuracy: "+str(Net.layers[1].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[1].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[1].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[1].svm_norm_hist_accuracy))

#%% Network test (only two layer like old times)



#First layer parameters
input_channels = 32 + 32*use_all_addr

df = pd.read_csv (r'whitenoise.csv')
mean_rate = np.asarray(df.columns[:], dtype=float)
channel_taus = 1/mean_rate

channel_taus = channel_taus/channel_taus[0]

tau_K = 500

taus = tau_K*channel_taus

n_features=20
local_tv_length=10
n_input_channels=input_channels
n_batch_files=None
dataset_runs=1

local_layer_parameters = [n_features, local_tv_length, n_input_channels, taus,\
                    n_batch_files, dataset_runs]

#Pool Layer
n_input_channels=32
pool_factor = 8
pool_layer_parameters = [n_input_channels, pool_factor]

#Second layer parameters
n_input_features=n_features 
n_input_channels=4
n_features=64
cross_tv_width=None

taus=50e3
n_batch_files=None
dataset_runs=1


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   
             
Net = GORDONN(n_threads=24, verbose=True, low_memory_mode=False)
Net.add_layer("Local", local_layer_parameters)
Net.add_layer("Pool", pool_layer_parameters)    
Net.add_layer("Cross", cross_layer_parameters)
Net.learn(dataset_train,labels_train,classes)
Net.predict(dataset_test, labels_train, labels_test, classes)

print("Histogram accuracy: "+str(Net.layers[0].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[0].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[0].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[0].svm_norm_hist_accuracy))


print("Histogram accuracy: "+str(Net.layers[2].hist_accuracy))
print("Norm Histogram accuracy: "+str(Net.layers[2].norm_hist_accuracy))
print("SVC Histogram accuracy: "+str(Net.layers[2].svm_hist_accuracy))
print("SVC norm Histogram accuracy: "+str(Net.layers[2].svm_norm_hist_accuracy))

#%% Last layer new classifier

#Actual decay
df = pd.read_csv (r'whitenoise.csv')
mean_rate =np.asarray(df.columns[:], dtype=float)
channel_taus = 1/mean_rate
channel_taus = channel_taus/channel_taus[0]

n_threads=24

#First Layer parameters
Tau_T=125
taus = (Tau_T*channel_taus)

input_channels = 32 + 32*use_all_addr
n_features=20
local_tv_length=10
n_input_channels=input_channels
n_batch_files=None
dataset_runs=1

local_layer_parameters = [n_features, local_tv_length, n_input_channels, taus,\
                    n_batch_files, dataset_runs]

Net = GORDONN(n_threads=n_threads, verbose=True, server_mode=False)
Net.add_layer("Local", local_layer_parameters)

#Second layer parameters
n_input_features=n_features 
n_input_channels=32
# n_features=64
n_features=32
cross_tv_width=3 
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)



#Pool Layer
n_input_channels=32
pool_factor=3
Net.add_layer("Pool", [n_input_channels, pool_factor])


Net.learn(dataset_train,labels_train,classes)
Net.predict(dataset_test, labels_train, labels_test, classes)

#%%
#Third layer parameters
n_input_features=n_features 
n_input_channels=16
n_input_channels=11
n_hidden_units=128
cross_tv_width=3 
taus=120e3
n_labels=number_of_labels
learning_rate=1e-4

from Libs.GORDONN.Layers.Cross_class_layer import Cross_class_layer


             
class_layer = Cross_class_layer(n_hidden_units, cross_tv_width, 
                                n_input_channels, taus, n_labels, learning_rate,
                                n_input_features, n_batch_files,
                                dataset_runs, n_threads, True)

class_layer.learn(Net.net_response_train[-1],labels_train)