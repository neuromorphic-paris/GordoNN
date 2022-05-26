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
import os, gc, pickle, copy
from tensorflow.keras.callbacks import EarlyStopping
from Libs.Solid_HOTS._General_Func import create_mlp
from joblib import Parallel, delayed 
from sklearn import svm
import pandas as pd

from tensorflow.keras.layers import Input, Dense, BatchNormalization,\
                                    Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers

# To use CPU for training
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# To allow more memory to be allocated on gpus incrementally
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
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
from Libs.GORDONN.Layers.MIG_Layer import MIG_Layer

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

local_layer_parameters = [n_features, local_tv_length, taus,\
                    n_batch_files, dataset_runs]

Net = GORDONN(n_threads=n_threads, verbose=True, server_mode=False, low_memory_mode=False)
Net.add_layer("Local", local_layer_parameters, n_input_channels)

#Second layer parameters
n_features=64
# n_features=32
cross_tv_width=3 
taus=20e3


cross_layer_parameters = [n_features, cross_tv_width, taus, n_batch_files,
                          dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

# Net.learn(dataset_train,labels_train,classes)
# Net.predict(dataset_test, labels_train, labels_test, classes)

#%% Further addictions 



#MIG Layers
MI_factor=99.99
Net.add_layer("MIG", [MI_factor])

#Pool Layer
pool_factor=2
Net.add_layer("Pool", [pool_factor])

#Third layer parameters
n_features=128
cross_tv_width=3
taus=1e6

cross_layer_parameters = [n_features, cross_tv_width, taus, n_batch_files,
                          dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

#MIG Layer
MI_factor=95
Net.add_layer("MIG", [MI_factor])

Net.learn(dataset_train,labels_train,classes)
Net.predict(dataset_test, labels_train, labels_test, classes)
#%%
Net.layers[5].MI_factor=95
Net.layers[4].cross_tv_width=5
Net.layers[4].n_features=64

Net.learn(dataset_train,labels_train,classes,rerun_layer=4)
Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=4)

#%% Add layers

#Pool Layer
pool_factor=2
Net.add_layer("Pool", [pool_factor])

#Third layer parameters
n_features=128
cross_tv_width=3 
taus=1e6

cross_layer_parameters = [n_features, cross_tv_width, taus, n_batch_files,
                          dataset_runs]   

Net.add_layer("Cross", cross_layer_parameters)

Net.learn(dataset_train,labels_train,classes,rerun_layer=5)
Net.predict(dataset_test, labels_train, labels_test, classes, rerun_layer=5)
#%%

rec = 10

n_layers = len(Net.architecture)
fig, axs = plt.subplots(1, n_layers)

for layer in range(n_layers):
    word = classes[labels_train[rec]]
    fig_title = "Word: '"+word+"' Layer: "+str(layer)+" "+Net.architecture[layer]
    timestamps = Net.net_response_train[layer][rec][0]
    channels = Net.net_response_train[layer][rec][1]
    features = Net.net_response_train[layer][rec][2]
    axs[layer].scatter(timestamps,channels, c=features)
    axs[layer].set_title(fig_title)
    axs[layer].set_ylabel("Channel Index")
    axs[layer].set_xlabel("Microseconds")


#%%
#Third layer parameters
n_input_features=n_features 
# n_input_channels=16
n_input_channels=11
n_hidden_units=128
cross_tv_width=3 
taus=120e3
n_labels=number_of_labels
learning_rate=1e-3
mlp_epochs=50
mlp_ts_batch_size=128

from Libs.GORDONN.Layers.Cross_class_layer import Cross_class_layer

             
class_layer = Cross_class_layer(n_hidden_units, cross_tv_width, taus, n_labels,
                                learning_rate, mlp_ts_batch_size, mlp_epochs,
                                n_input_channels, n_input_features,
                                n_batch_files, dataset_runs, n_threads, True)

class_layer.learn(Net.net_response_train[-1],labels_train)
mlp = class_layer.mlp
#%%

train_tv, train_labels_one_hot = class_layer.tv_generation(Net.net_response_train[-1],labels_train)
test_tv, test_labels_one_hot = class_layer.tv_generation(Net.net_response_test[-1],labels_test)

#%%% Save
with open('train_tv.npy', 'wb') as f:
    np.save(f, train_tv)
with open('train_labels_one_hot.npy', 'wb') as f:
    np.save(f, train_labels_one_hot)
with open('test_tv.npy', 'wb') as f:
    np.save(f, test_tv)
with open('test_labels_one_hot.npy', 'wb') as f:
    np.save(f, test_labels_one_hot)
with open('labels_train.npy', 'wb') as f:
    np.save(f, labels_train)
with open('labels_test.npy', 'wb') as f:
    np.save(f, labels_test)
# with open('dataset_train.npy', 'wb') as f:
#     np.save(f, dataset_train)
# with open('dataset_test.npy', 'wb') as f:
#     np.save(f, dataset_test)    
#%%% Load
with open('train_tv.npy', 'rb') as f:
    train_tv=np.load(f)
with open('train_labels_one_hot.npy', 'rb') as f:
    train_labels_one_hot=np.load(f)
with open('test_tv.npy', 'rb') as f:
    test_tv=np.load(f)
with open('test_labels_one_hot.npy', 'rb') as f:
    test_labels_one_hot=np.load(f)
with open('labels_train.npy', 'rb') as f:
    labels_train=np.load(f)
with open('labels_test.npy', 'rb') as f:
    labels_test=np.load(f)
# with open('dataset_train.npy', 'rb') as f:
#     dataset_train=np.load(f)
# with open('dataset_test.npy', 'rb') as f:
#     dataset_test=np.load(f)
#%%

#Third layer parameters
n_input_features=32 
# n_input_channels=16
n_input_channels=11
n_hidden_units=128
cross_tv_width=3 
taus=120e36
n_labels=number_of_labels
learning_rate=1e-3
mlp_epochs=50
mlp_ts_batch_size=128

mlp = create_mlp(n_input_features*cross_tv_width, n_hidden_units, n_labels, learning_rate)
mlp.fit(train_tv, train_labels_one_hot, epochs=mlp_epochs, batch_size=mlp_ts_batch_size, 
        shuffle=True)
#%% Test the code 
pred_test=mlp.predict(test_tv)

ev_count = 0
n_test_recs = len(dataset_test)
responses = np.zeros([n_test_recs, len(classes)])
for test_rec_i in range(n_test_recs):
    n_ev = len(dataset_test[test_rec_i][0])
    responses[test_rec_i] = sum(pred_test[ev_count:ev_count+n_ev,:])
    
predicted_labels = np.argmax(responses,1)
percent_correct=100*sum((labels_test-predicted_labels)==0)/n_test_recs
#%%

# =============================================================================
def create_mlp(input_size, hidden_size, output_size, learning_rate):
    """
    Function used to create a small mlp used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        mlp (keras model) : the freshly baked network
    """
    def relu_advanced(x):
        return K.activations.relu(x, alpha=0.3)
    
    inputs = Input(shape=(input_size,), name='encoder_input')
    # x = BatchNormalization()(inputs)
    x = Dense(hidden_size, activation='sigmoid')(inputs)
    # x = Dropout(0.3)(x)#0.3
    x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.7)(x)#0.7
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.9)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)
    outputs = Dense(output_size, activation='sigmoid')(x)
    
    
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    # mlp.compile(optimizer=adam,
    #           loss='categorical_crossentropy', metrics=['accuracy'])
    mlp.compile(optimizer=adam,
              loss='MSE', metrics=['accuracy'])    
    return mlp      

#%% Net Mutual info layer 2 
i_layer = 4
MI_factor = 95

n_recordings = len(Net.net_response_train[i_layer])
n_clusters = len(Net.layers[i_layer].features)
labels = labels_train
activity = np.zeros([n_recordings, n_clusters])
for recording in range(n_recordings): 
    clusters = np.unique(Net.net_response_train[i_layer][recording][-1])
    activity[recording, clusters]=1;

p_r_s = np.zeros([number_of_labels, n_clusters])
p_s = 1/number_of_labels
p_r = np.sum(activity,axis=0)/(n_recordings)
p_not_r = 1-p_r

for label in range(number_of_labels):
    indx = labels==label
    p_r_s[label]+=np.sum(activity[indx,:],axis=0)/sum(labels==label)

p_not_r_s = 1-p_r_s

MI_cluster_on = p_r_s*np.log2(p_r_s/((p_r)+np.logical_not(p_r)) + np.logical_not(p_r_s))
MI_cluster_off = p_not_r_s*np.log2(p_not_r_s/((p_not_r)+np.logical_not(p_not_r)) + np.logical_not(p_not_r_s))

MI = p_s*(MI_cluster_on+MI_cluster_off)

featMI = np.sum(MI, axis=0)
sortfeats = np.argsort(featMI)[::-1]
featMI = featMI[sortfeats]
cumMI = (np.cumsum(featMI)/sum(featMI))*100

topfeats = sortfeats[cumMI<=MI_factor]

plt.figure()
plt.plot(np.cumsum(featMI)/sum(featMI))
plt.ylabel("Bits of information (over 3)")
plt.xlabel("#Features")
plt.title("Sorted MI per cluster")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

# topclusters = np.argsort(clusterMI)[-150:]
rel_clusterMI = featMI/(1+0.01*np.arange(n_clusters))


plt.figure()
plt.plot(rel_clusterMI/n_clusters)
plt.ylabel("Bits of information (over 3)/#Features")
plt.xlabel("#Features")
plt.title("Relative sorted MI")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

#%%  MI calculation 
    
n_clusters = len(featMI)


# The histograms for each class, also known as "signatures"     
signatures = Net.layers[i_layer].sign[:,:,sortfeats]
signatures_norm = Net.layers[i_layer].norm_sign[:,:,sortfeats]

hist_train = Net.layers[i_layer].hists[:,:,sortfeats]
norm_hist_train = Net.layers[i_layer].norm_hists[:,:,sortfeats]

hist_test = Net.layers[i_layer].hists_test[:,:,sortfeats]
norm_hist_test = Net.layers[i_layer].norm_hists_test[:,:,sortfeats]

accuracy=np.zeros(n_clusters)
norm_accuracy=np.zeros(n_clusters)

for cluster_i in range(n_clusters):
    for recording in range(len(hist_train)):
        
        closest_file=np.argmin(np.sum((hist_train[recording,:,:cluster_i+1]-signatures[:,:,:cluster_i+1])**2, axis=(1,2)))
        closest_file_norm=np.argmin(np.sum((norm_hist_train[recording,:,:cluster_i+1]-signatures_norm[:,:,:cluster_i+1])**2, axis=(1,2)))

    
        label = labels_train[recording]
        if closest_file==label:
            accuracy[cluster_i]+=1/len(hist_train)
       
        if closest_file_norm==label:
            norm_accuracy[cluster_i]+=1/len(hist_train)
    
accuracy_test=np.zeros(n_clusters)
norm_accuracy_test=np.zeros(n_clusters)

for cluster_i in range(n_clusters):
    for recording in range(len(hist_test)):
        
        closest_file=np.argmin(np.sum((hist_test[recording,:,:cluster_i+1]-signatures[:,:,:cluster_i+1])**2, axis=(1,2)))
        closest_file_norm=np.argmin(np.sum((norm_hist_test[recording,:,:cluster_i+1]-signatures_norm[:,:,:cluster_i+1])**2, axis=(1,2)))
    
        label = labels_test[recording]
        if closest_file==label:
            accuracy_test[cluster_i]+=1/len(hist_test)
       
        if closest_file_norm==label:
            norm_accuracy_test[cluster_i]+=1/len(hist_test)

# print("Accuracy is; "+str(accuracy*100))
# print("Accuracy of normalized signatures is; "+str(norm_accuracy*100))

plt.figure()
plt.plot(accuracy)
plt.ylabel("% Correct recordings")
plt.title("Accuracy by MI Gating")
plt.xlabel("#Features")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

plt.figure()
plt.plot(accuracy_test)
plt.ylabel("% Correct recordings")
plt.title("Test Accuracy by MI Gating")
plt.xlabel("#Features")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

plt.figure()
plt.plot(norm_accuracy)
plt.ylabel("% Correct recordings")
plt.title("Norm Accuracy by MI Gating")
plt.xlabel("#Features")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

plt.figure()
plt.plot(norm_accuracy_test)
plt.ylabel("% Correct recordings")
plt.title("Norm Test Accuracy by MI Gating")
plt.xlabel("#Features")
plt.vlines(len(topfeats),plt.ylim()[0],plt.ylim()[1])

#%% Remove events

data = Net.net_response_train[-1]

def feature_events_pruning(rec_data, feature_idxs):
    """
    This function is used to remove all events in a recording that have an 
    index equal to indexes present in feature_idxs (a list of indexes).
    It returns the same rec data with a new set reassigned indexes for the
    remaining events (to avoid missing cluster indexes)
    """
    
    n_data_dim = len(rec_data)
    f_index_data = rec_data[-1]
    new_f_index_data = copy.deepcopy(f_index_data)
    
    for f_index in feature_idxs:
        new_f_index_data[f_index_data==f_index]=-1
        new_f_index_data[new_f_index_data>f_index]--1
        feature_idxs[feature_idxs>f_index]--1
    
    relevant_ev_indxs = new_f_index_data>-1
    
    new_rec_data = []
    for data_dim in range(n_data_dim-1):
        new_rec_data.append(rec_data[data_dim][relevant_ev_indxs])
    
    new_rec_data.append(new_f_index_data)    
    
    return new_rec_data
