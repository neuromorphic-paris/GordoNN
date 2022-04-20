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
import pandas as pd


# To use CPU for training
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# To allow more memory to be allocated on gpus incrementally
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
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


#Original
channel_taus = np.linspace(2,9,32)
channel_taus = channel_taus/channel_taus[0]

#No Att
channel_taus = 1


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
layer_res = {'Eucl_res': eucl_res, 'Norm_eucl_res': euclnorm_res,\
             'svc_Eucl_res': svc_eucl_res, 'svc_Norm_eucl_res': svc_euclnorm_res,\
             'Taus_T' : Tau_T_first}

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
