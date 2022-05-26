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
import pandas as pd


# To use CPU for training
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""log

# To allow more memory to be allocated on gpus incrementally
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load, data_load

from Libs.GORDONN.Layers.Local_Layer import Local_Layer
from Libs.GORDONN.Layers.Cross_Layer import Cross_Layer
from Libs.GORDONN.Layers.Pool_Layer import Pool_Layer
from Libs.GORDONN.Network import GORDONN

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

#Actual decay
df = pd.read_csv (r'whitenoise.csv')
mean_rate =np.asarray(df.columns[:], dtype=float)
channel_taus = 1/mean_rate
channel_taus = channel_taus/channel_taus[0]

input_channels = 32 + 32*use_all_addr
n_features=20
local_tv_length=10
n_input_channels=input_channels
n_batch_files=None
dataset_runs=1
n_threads=24

#Original
# channel_taus = np.linspace(2,9,32)
# channel_taus = channel_taus/channel_taus[0]

#No Att
# channel_taus = 1


#%% First layer decay search

Tau_T_first = np.power(10,np.arange(0.1,4,0.3))*2

eucl_res= []
euclnorm_res = []
svc_eucl_res=[]
svc_euclnorm_res = []

for Tau_T in Tau_T_first:
    
    taus = (Tau_T*channel_taus)
    
    layer_parameters = [n_features, local_tv_length, n_input_channels, taus,\
                        n_batch_files, dataset_runs]
                    
    Net = GORDONN(n_threads=24, verbose=True)
    Net.add_layer("Local", layer_parameters)
    Net.learn(dataset_train,labels_train,classes)
    Net.predict(dataset_test, labels_train, labels_test, classes)
    
    print("Histogram accuracy: "+str(Net.layers[0].hist_accuracy))
    print("Norm Histogram accuracy: "+str(Net.layers[0].norm_hist_accuracy))
    print("SVC Histogram accuracy: "+str(Net.layers[0].svm_hist_accuracy))
    print("SVC norm Histogram accuracy: "+str(Net.layers[0].svm_norm_hist_accuracy))


    eucl_res.append(Net.layers[0].hist_accuracy)
    euclnorm_res.append(Net.layers[0].norm_hist_accuracy)
    svc_eucl_res.append(Net.layers[0].svm_hist_accuracy)
    svc_euclnorm_res.append(Net.layers[0].svm_norm_hist_accuracy)

#%% Save Layer results
# layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res,\
#              'svc_Eucl_res': svc_eucl_res, 'svc_Norm_eucl_res': svc_euclnorm_res,\
#              'Taus_T' : Tau_T_first}

# with open('Results/Decay_search_tmp/no_att.pickle', 'wb') as handle:
#     pickle.dump(layer_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# #%% Load Layer results
filename = "no_att"
with open('Results/Decay_search_tmp/'+str(filename)+'.pickle', 'rb') as handle:
    layer_res = pickle.load(handle)


#%% Load Layer results
filename_old = "model_actual"
with open('Results/Decay_search_tmp/'+str(filename_old)+'.pickle', 'rb') as handle:
    layer_res_old = pickle.load(handle)

#%% Comparison HOTS layer and Hack Layer norm eucl
plt.figure()
plt.plot(layer_res_old['Taus_T'], layer_res_old['Norm_eucl_res'], label=filename_old)
plt.plot(layer_res['Taus_T'], layer_res['Norm_eucl_res'], label=filename)
plt.xlabel("Tau first layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Parameter search simple Euclidean classifier")
plt.legend()

#%% Comparison HOTS layer and Hack Layer eucl
plt.figure()
plt.plot(layer_res_old['Taus_T'], layer_res_old['Eucl_res'], label=filename_old)
plt.plot(layer_res['Taus_T'], layer_res['Eucl_res'], label=filename)
plt.xlabel("Tau first layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Parameter search simple normalized Euclidean classifier")
plt.legend()

#%% Comparison HOTS layer and Hack Layer svc 
plt.figure()
plt.plot(layer_res_old['Taus_T'], layer_res_old['svc_Eucl_res'], label=filename_old)
plt.plot(layer_res['Taus_T'], layer_res['svc_Eucl_res'], label=filename)
plt.xlabel("Tau first layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Parameter search SVM classifier")
plt.legend()

#%% Comparison HOTS layer and Hack Layer svc norm
plt.figure()
plt.plot(layer_res_old['Taus_T'], layer_res_old['svc_Norm_eucl_res'], label=filename_old)
plt.plot(layer_res['Taus_T'], layer_res['svc_Norm_eucl_res'], label=filename)
plt.xlabel("Tau first layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Parameter search SVM normalized classifier")
plt.legend()

#%% Second layer decay search

Tau_T=125
taus = (Tau_T*channel_taus)

local_layer_parameters = [n_features, local_tv_length, n_input_channels, taus,\
                    n_batch_files, dataset_runs]

Net = GORDONN(n_threads=24, verbose=True)
Net.add_layer("Local", local_layer_parameters)
Net.learn(dataset_train, labels_train, classes)
Net.predict(dataset_test, labels_train, labels_test, classes)

#Second layer parameters
n_input_features=n_features 
n_input_channels=input_channels
n_features=64
# cross_tv_width=6 
cross_tv_width=3 
taus=50e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

# Tau_C_first = np.power(10,np.arange(0.1,5,0.3))*2

# Tau_C_first = np.power(10,np.arange(4,4.8,0.05))*2

Tau_C_first = np.power(10,np.arange(0.1,4.8,0.1))*20

eucl_res= []
euclnorm_res = []
svc_eucl_res=[]
svc_euclnorm_res = []


for Tau_indx, Tau_C in enumerate(Tau_C_first):
    
    Net.layers[1].taus=Tau_C
    if Tau_indx==0:
        Net.learn(dataset_train, labels_train, classes)
        Net.predict(dataset_test, labels_train, labels_test, classes)
    else:        
        Net.learn(dataset_train, labels_train, classes, rerun_layer=1)
        Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=1)
    
    print("Histogram accuracy: "+str(Net.layers[1].hist_accuracy))
    print("Norm Histogram accuracy: "+str(Net.layers[1].norm_hist_accuracy))
    print("SVC Histogram accuracy: "+str(Net.layers[1].svm_hist_accuracy))
    print("SVC norm Histogram accuracy: "+str(Net.layers[1].svm_norm_hist_accuracy))


    eucl_res.append(Net.layers[1].hist_accuracy)
    euclnorm_res.append(Net.layers[1].norm_hist_accuracy)
    svc_eucl_res.append(Net.layers[1].svm_hist_accuracy)
    svc_euclnorm_res.append(Net.layers[1].svm_norm_hist_accuracy)

#%% Save Layer results
layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res,\
              'svc_Eucl_res': svc_eucl_res, 'svc_Norm_eucl_res': svc_euclnorm_res,\
              'Taus_C' : Tau_C_first}

with open('Results/Decay_search_tmp/Lay_2_small.pickle', 'wb') as handle:
    pickle.dump(layer_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% Load Layer results
filename = "Lay_2_small"
with open('Results/Decay_search_tmp/'+str(filename)+'.pickle', 'rb') as handle:
    layer_res = pickle.load(handle)

#%% Plot layer 2 results

plt.plot(layer_res['Taus_C'], layer_res['Eucl_res'], label="Eucl_res")
plt.plot(layer_res['Taus_C'], layer_res['Norm_eucl_res'], label="Norm_eucl_res")
plt.plot(layer_res['Taus_C'], layer_res['svc_Eucl_res'], label="svc_Eucl_res")
plt.plot(layer_res['Taus_C'], layer_res['svc_Norm_eucl_res'], label="svc_Norm_eucl_res")

plt.xlabel("Tau second layer (us)")
plt.ylabel("Recognition rates")
plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
plt.title("Classifier performance")
plt.legend()

#%% Fourth (Second cross layer) layer decay search

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

Net = GORDONN(n_threads=24, verbose=True)
Net.add_layer("Local", local_layer_parameters)

#Second layer parameters
n_input_features=n_features 
n_input_channels=32
n_features=64
cross_tv_width=3 
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)



#Pool Layer
n_input_channels=32
pool_factor=2
Net.add_layer("Pool", [n_input_channels, pool_factor])


#4th Cross layer
n_input_features=n_features 
n_input_channels=16
n_features=128
cross_tv_width=3 
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)


Tau_C_first = np.power(10,np.arange(0.1,4.8,0.1))*20

eucl_res= []
euclnorm_res = []
svc_eucl_res=[]
svc_euclnorm_res = []


for Tau_indx, Tau_C in enumerate(Tau_C_first):
    
    Net.layers[3].taus=Tau_C
    if Tau_indx==0:
        Net.learn(dataset_train, labels_train, classes)
        Net.predict(dataset_test, labels_train, labels_test, classes)
    else:  
        Net.learn(dataset_train, labels_train, classes, rerun_layer=3)
        Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=3)
    
    print("Histogram accuracy: "+str(Net.layers[3].hist_accuracy))
    print("Norm Histogram accuracy: "+str(Net.layers[3].norm_hist_accuracy))
    print("SVC Histogram accuracy: "+str(Net.layers[3].svm_hist_accuracy))
    print("SVC norm Histogram accuracy: "+str(Net.layers[3].svm_norm_hist_accuracy))


    eucl_res.append(Net.layers[3].hist_accuracy)
    euclnorm_res.append(Net.layers[3].norm_hist_accuracy)
    svc_eucl_res.append(Net.layers[3].svm_hist_accuracy)
    svc_euclnorm_res.append(Net.layers[3].svm_norm_hist_accuracy)

#%% Save Layer results
# layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res,\
#               'svc_Eucl_res': svc_eucl_res, 'svc_Norm_eucl_res': svc_euclnorm_res,\
#               'Taus_C' : Tau_C_first}

# with open('Results/Decay_search_tmp/Lay_4.pickle', 'wb') as handle:
#     pickle.dump(layer_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% Load Layer results
# filename = "Lay_4"
# with open('Results/Decay_search_tmp/'+str(filename)+'.pickle', 'rb') as handle:
#     layer_res = pickle.load(handle)

#%% Plot layer 4 results

# plt.plot(layer_res['Taus_C'], layer_res['Eucl_res'], label="Eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['Norm_eucl_res'], label="Norm_eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['svc_Eucl_res'], label="svc_Eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['svc_Norm_eucl_res'], label="svc_Norm_eucl_res")

# plt.xlabel("Tau second layer (us)")
# plt.ylabel("Recognition rates")
# plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
# plt.title("Classifier performance")
# plt.legend()


#%% Sixth (3th cross layer) layer decay search

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

Net = GORDONN(n_threads=n_threads, verbose=True, server_mode=True)
Net.add_layer("Local", local_layer_parameters)

#Second layer parameters
n_input_features=n_features 
n_input_channels=32
n_features=64
# n_features=32
cross_tv_width=3 
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)



#Pool Layer
n_input_channels=32
pool_factor=2
Net.add_layer("Pool", [n_input_channels, pool_factor])


#Third layer parameters
n_input_features=n_features 
n_input_channels=16
n_features=128
# n_features=64
cross_tv_width=5 
taus=1e6


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

#Pool Layer
n_input_channels=16
pool_factor = 4
Net.add_layer("Pool", [n_input_channels, pool_factor])

#6th Cross layer
n_input_features=n_features 
n_input_channels=4
n_features=256
cross_tv_width=None
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, 
                    n_input_channels, taus, 
                    n_input_features, n_batch_files,
                    dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

Tau_C_first = np.power(10,np.arange(0.1,4.8,0.1))*20

eucl_res= []
euclnorm_res = []
svc_eucl_res=[]
svc_euclnorm_res = []


for Tau_indx, Tau_C in enumerate(Tau_C_first):
    
    Net.layers[5].taus=Tau_C
    if Tau_indx==0:
        Net.learn(dataset_train, labels_train, classes)
        Net.predict(dataset_test, labels_train, labels_test, classes)
    else:  
        Net.learn(dataset_train, labels_train, classes, rerun_layer=5)
        Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=5)
    
    print("Histogram accuracy: "+str(Net.layers[5].hist_accuracy))
    print("Norm Histogram accuracy: "+str(Net.layers[5].norm_hist_accuracy))
    print("SVC Histogram accuracy: "+str(Net.layers[5].svm_hist_accuracy))
    print("SVC norm Histogram accuracy: "+str(Net.layers[5].svm_norm_hist_accuracy))


    eucl_res.append(Net.layers[5].hist_accuracy)
    euclnorm_res.append(Net.layers[5].norm_hist_accuracy)
    svc_eucl_res.append(Net.layers[5].svm_hist_accuracy)
    svc_euclnorm_res.append(Net.layers[5].svm_norm_hist_accuracy)

#%% Save Layer results
layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res,\
              'svc_Eucl_res': svc_eucl_res, 'svc_Norm_eucl_res': svc_euclnorm_res,\
              'Taus_C' : Tau_C_first}

with open('Results/Decay_search_tmp/Lay_6_final.pickle', 'wb') as handle:
    pickle.dump(layer_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% Load Layer results
# filename = "Lay_6"
# with open('Results/Decay_search_tmp/'+str(filename)+'.pickle', 'rb') as handle:
#     layer_res = pickle.load(handle)

#%% Plot layer 6 results

# plt.plot(layer_res['Taus_C'], layer_res['Eucl_res'], label="Eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['Norm_eucl_res'], label="Norm_eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['svc_Eucl_res'], label="svc_Eucl_res")
# plt.plot(layer_res['Taus_C'], layer_res['svc_Norm_eucl_res'], label="svc_Norm_eucl_res")

# plt.xlabel("Tau second layer (us)")
# plt.ylabel("Recognition rates")
# plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
# plt.title("Classifier performance")
# plt.legend()