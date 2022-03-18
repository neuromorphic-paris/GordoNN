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
plt.style.use("dark_background")

### Selecting the dataset
shuffle_seed = 25 # seed used for dataset shuffling if set to 0 the process will be totally random 
# shuffle_seed = 0 

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

# number_files_dataset = 3740
number_files_dataset = 500
train_test_ratio = 0.70
use_all_addr = False
number_of_labels = 2
dataset_folder = "Data/On_Off"
spacing = 1


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



[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, classes] = data_load(dataset_folder, number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr, spacing)


## IF YOU WANT TO SAVE A PRECISE SET OF FILENAMES IN CASE THE SEED IS NOT RELIABLE AMONG DIFFERENT COMPUTERS
#np.save("filenames_train",filenames_train)
#np.save("filenames_test",filenames_test)
#np.save("labels_train",labels_train)
#np.save("labels_test",labels_test)


#%% Genre Dataset
# 10 class of recordings are used. blues, classical, country, disco, hip-hop, jazz
# metal, pop, reggae, rock. 100 files per genre except jazz (99)
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class 
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channels of the cochlea
# =============================================================================

number_files_dataset = 99
train_test_ratio = 0.70
use_all_addr = False
number_of_labels = 10
dataset_folder ='Data/Genres'
spacing = 1


[dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, classes] = data_load(dataset_folder, number_files_dataset, train_test_ratio, shuffle_seed, use_all_addr, spacing)

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


# number_files_dataset = 1712
number_files_dataset = 1000
# number_files_dataset = 50

# train_test_ratio = 0.70
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


# features_number=[[6,96]] 
# features_number=[[6,256]] 
features_number=[[20,256] ] 

# features_number=[[6,32],[1,64],[1,64],[1,64],[1,96]] 
# local_surface_lengths = [5,1,1,1,1]
# local_surface_lengths = [5,1]
local_surface_lengths = [5,1]


input_channels = 32 + 32*use_all_addr

### Channel Taus ###

##Theorical _Probably Wrong
#channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
#                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
#                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 

## Uniform
#channel_taus = np.arange(32)*1

#Linear interpolation between highest spike frequency 90ks/s to lowest 20ks/s, used to balance the filters
channel_taus = np.linspace(2,9,32)
                                                             
# taus_T_coeff = np.array([1000,1,1,1,1]) # Multiplicative coefficients to help to change quickly the taus_T  #1000
# taus_T = (taus_T_coeff*[channel_taus,np.ones(32),np.ones(64),np.ones(64),np.ones(96)]).tolist()
# taus_2D = [10000,20000,40000,80000,100000]  


# taus_T_coeff = np.array([1000,1]) # Multiplicative coefficients to help to change quickly the taus_T  #1000
taus_T_coeff = np.array([1500,1]) # Multiplicative coefficients to help to change quickly the taus_T  #1000
taus_T = (taus_T_coeff*[channel_taus,np.ones(256)]).tolist()
taus_2D = [100000]  

# taus_2D = [taus_T[0], 100000]  

#n_batch_files = 128
# n_batch_files = 2048
n_batch_files = 512


# dataset_runs = 10 #how many times the dataset is run for clustering
dataset_runs = 25 #how many times the dataset is run for clustering



threads=24 
          
verbose=True

network_parameters = [[features_number, local_surface_lengths, input_channels, taus_T, taus_2D, 
                 threads, verbose],[n_batch_files, dataset_runs]]


#%% Network creation and learning 

# Create the network
Net = Solid_HOTS_Net(network_parameters)

    
# Learn the features
Net.learn(dataset_train)
Net.infer(dataset_test)


#%% Network creation and learning 

# Create the network
Net = Solid_HOTS_Net(network_parameters)

Net.cross_surface_width = [5,1]

    
# Learn the features
Net.learn(dataset_train)
Net.infer(dataset_test)



#%% Net Mutual info
n_recordings = len(Net.last_layer_activity)
n_clusters = Net.features_number[-1][-1]
labels = labels_train
activity = np.zeros([n_recordings, n_clusters])
for recording in range(n_recordings): 
    clusters = np.unique(Net.last_layer_activity[recording][1])
    activity[recording, clusters]=1;

p_r_s = np.zeros([number_of_labels, n_clusters])
p_s = 1/number_of_labels
p_r = np.sum(activity,axis=0)/(n_recordings)
for label in range(number_of_labels):
    indx = labels==label
    p_r_s[label]+=np.sum(activity[indx,:],axis=0)/sum(labels_train==label)

#number of recordings is the same per each label.
# p_r = p_r/(n_recordings*number_of_labels)

IndividualMI = p_s*p_r_s*np.log2(p_r_s/((p_r)+np.logical_not(p_r)) + np.logical_not(p_r_s))
clusterMI = np.sum(IndividualMI, axis=0)
topclusters = np.argsort(clusterMI)
sortedrelClusterMI = clusterMI[topclusters]#/sum(clusterMI)
plt.figure()
plt.plot(np.cumsum(sortedrelClusterMI[::-1]))
plt.ylabel("Bits of information (over 3)")
# topclusters = np.argsort(clusterMI)[-150:]


    
#%% Look at the activations per per label

data =  np.array(Net.last_layer_activity_test)
plt.figure()
plt.title("Activations per class")
plt.xlabel("Time us")
plt.ylabel("Last layer cluster activations")

label_i = 0
indx = np.where(labels_test==label_i)[0]
times = data[indx][:,0].tolist()
centers = data[indx][:,1].tolist()

for recording in range(1):
   plt.plot(times[recording], centers[recording], color='r')


plt.figure()
plt.title("Activations per class")
plt.xlabel("Time us")
plt.ylabel("Last layer cluster activations")

   
label_i = 1
indx = np.where(labels_test==label_i)[0]
times = data[indx][:,0].tolist()
centers = data[indx][:,1].tolist()

for recording in range(1):
   plt.plot(times[recording], centers[recording], color='g')

#%%
Net.features_number=features_number
Net.polarities=[32,96,128,256,128]
Net.taus_T = taus_T
Net.learn(dataset_train, rerun_layer=2)
Net.infer(dataset_test, rerun_layer=2)
#%% Methods to add layers/rerun training

# # If you want to add a layer or more on top of the net and keep the results
# # you had for the first ones use this. 
# # You will have to load new parameters and run the learning from the 
# # index of the new layer (You computed a 2 layer network, you want to add 2 more,
# # you use a new set of parameters, and rerun learning with rerun_layer=2, as the 
# # layers index start with 0.
# Net.add_layers(network_parameters)
# Net.lear;.9i´n(dataset_train, dataset_test, rerun_layer = 2)

# # If you want to recompute few layers of the net in a sequential manner
# # change and load the parameters with this method and then rerun the net learning
# # (For example you computed a 2 layer network, and you want to rerun the second layer,
# # you use a new set of parameters, and rerun learning with rerun_layer=1, as the 
# # layers index start with 0.
# Net.load_parameters(network_parameters)
# Net.learn(dataset_train, dataset_test, rerun_layer = 1)


# #%% LSTM classifier training
# # Simple LSTM applied on all output events, binned in last_bin_width-size windows.
# gc.collect()

# lstm_bin_width = 200
# lstm_sliding_amount = 100
# lstm_units = 20
# lstm_learning_rate = 7e-5
# lstm_epochs = 5000
# lstm_batch_size = 128
# lstm_patience = 5000

# """
# bin_width = lstm_bin_width
# sliding_amount = lstm_sliding_amount
# units = lstm_units
# learning_rate = lstm_learning_rate
# epochs = lstm_epochs
# batch_size = lstm_batch_size
# patience = lstm_patience
# """


# Net.lstm_classification_train(dataset_train, labels_train, dataset_test, labels_test, number_of_labels, lstm_bin_width, 
#                               lstm_sliding_amount, lstm_learning_rate, lstm_units, lstm_epochs, 
#                               lstm_batch_size, lstm_patience)
# gc.collect()

# #%% LSTM test
# threshold = 0.5
# Net.lstm_classification_test(dataset_train, labels_train, dataset_test, labels_test, number_of_labels, lstm_bin_width, 
#                              lstm_sliding_amount, lstm_batch_size, threshold )



# #%% CNN classifier training

# learning_rate = 1e-4
# epochs = 15000
# batch_size = 64
# patience = 50
# bin_size = 20000


# Net.cnn_classification_train(dataset_train, labels_train, dataset_test, labels_test, number_of_labels, learning_rate,
#                              epochs, batch_size, bin_size, patience)
# gc.collect()


# #%% CNN classifier test

# batch_size = 64
# bin_size = 20000

# Net.cnn_classification_test(dataset_train, labels_train, dataset_test, labels_test, number_of_labels, batch_size, bin_size)


#%% Mlp classifier training
# Simple MLP applied on all output events as a weak classifier to prove HOTS
# working.

mlp_learning_rate = 1e-4
mlp_epochs = 20
mlp_hidden_size = 10
mlp_batch_size = 200000
patience = 500
number_of_labels=4

Net.mlp_classification_train(labels_train, labels_test, number_of_labels, mlp_learning_rate,
                              mlp_hidden_size, mlp_epochs, mlp_batch_size, patience)
gc.collect()

#%% Mlp classifier testing
threshold = 0.5
Net.mlp_classification_test(labels_test, number_of_labels, mlp_batch_size,
                            threshold)

#%% Histogram mlp classifier training
# Simple MLP applied over the histogram (the summed response of the last layer
# for each recording) of the net activity.

hist_mlp_learning_rate = 9e-5
hist_mlp_epochs = 20000
hist_mlp_hidden_size = 64
# hist_mlp_batch_size = 128
hist_mlp_batch_size = 2048
patience = 500

Net.hist_mlp_classification_train(labels_train, labels_test, number_of_labels, 
                                  hist_mlp_learning_rate, hist_mlp_hidden_size, 
                                  hist_mlp_epochs, hist_mlp_batch_size, patience)
gc.collect()

#%% Mlp hist classifier testing
threshold=0.5
Net.hist_mlp_classification_test(labels_test, number_of_labels,
                                 hist_mlp_batch_size, threshold)

#%% hist classifier testing
#training error
Net.hist_classification_test(labels_train, labels_test, number_of_labels)
                        
#%% MI histograms:

# Exctracting last layer activity         
last_layer_activity = Net.last_layer_activity.copy()
last_layer_activity_test = Net.last_layer_activity_test.copy()
num_of_recordings=len(last_layer_activity)
num_of_recordings_test=len(last_layer_activity_test)
    
n_clusters = len(topclusters)
# The histograms for each class, also known as "signatures"     
signatures = np.zeros([number_of_labels,n_clusters])
cl_hist_test = np.zeros([num_of_recordings_test,n_clusters])

signatures_norm = np.zeros([number_of_labels,n_clusters])
cl_hist_test_norm = np.zeros([num_of_recordings_test,n_clusters])

def cluster_counter(rec_activity, cluster_i):
    return sum(rec_activity==cluster_i)
    

for recording in range(num_of_recordings):
    rec_activity = last_layer_activity[recording][1]    
    label = labels_train[recording]
    rec_hist = Parallel(n_jobs=22)(delayed(cluster_counter)(rec_activity, cluster_i)\
                                   for cluster_i in topclusters)
    # rec_hist = [sum(rec_activity==cluster_i) for cluster_i in topclusters]
    signatures[label] += rec_hist
    n_events = (sum(rec_hist)+ (not sum(rec_hist)))
    signatures_norm[label] += rec_hist/n_events
    
for label in range(number_of_labels):
    signatures[label]=signatures[label]/sum(labels_train==label)
    signatures_norm[label]=signatures_norm[label]/sum(labels_train==label)

for recording in range(num_of_recordings_test):
    rec_activity = last_layer_activity_test[recording][1]    
    # rec_hist = [sum(rec_activity==cluster_i) for cluster_i in topclusters]
    rec_hist = Parallel(n_jobs=22)(delayed(cluster_counter)(rec_activity, cluster_i)\
                                   for cluster_i in topclusters)
    cl_hist_test[recording] = rec_hist
    n_events = (sum(rec_hist)+ (not sum(rec_hist)))
    cl_hist_test_norm[recording] = rec_hist/n_events

accuracy=np.zeros(n_clusters)
norm_accuracy=np.zeros(n_clusters)

for cluster_i in range(n_clusters):
    for recording in range(num_of_recordings_test):
        eucl_distance=np.linalg.norm(signatures[:,-(cluster_i+1):]-cl_hist_test[recording,-(cluster_i+1):],axis=1)
        eucl_distance_norm=np.linalg.norm(signatures_norm[:,-(cluster_i+1):]-cl_hist_test_norm[recording,-(cluster_i+1):],axis=1)
        closest_file = np.argmin(eucl_distance)
        closest_file_norm = np.argmin(eucl_distance_norm)
    
        label = labels_test[recording]
        if closest_file==label:
            accuracy[cluster_i]+=1/num_of_recordings_test
       
        if closest_file_norm==label:
                norm_accuracy[cluster_i]+=1/num_of_recordings_test
    

# print("Accuracy is; "+str(accuracy*100))
# print("Accuracy of normalized signatures is; "+str(norm_accuracy*100))

#%% Time based classifier: 

n_clusters = 100    
    
#Train    
response,ts = Net.cross_batch_tsMI_infering(Net.net_local_response, layer=0, clusters_ind=topclusters[-n_clusters:])
ts_labels = [labels_train[recording]*np.ones(len(ts[recording])) for recording in range(len(labels_train))]
svc = svm.SVC(decision_function_shape='ovr', kernel='poly')
ts=np.concatenate(ts)[::20]
ts_labels=np.concatenate(ts_labels)[::20]
# svc = svm.LinearSVC(multi_class='ovr')
svc.fit(ts, ts_labels)

#Test
response,ts = Net.cross_batch_tsMI_infering(Net.net_local_response_test, layer=0, clusters_ind=topclusters[-n_clusters:])
ts_labels = [labels_test[recording]*np.ones(len(ts[recording])) for recording in range(len(labels_test))]

accuracy_time=0
for recording in range(len(labels_test)):
    if ts[recording].size != 0:#check if it is not empty
        pred_labels_test = svc.predict(ts[recording])
        vals,counts = np.unique(pred_labels_test, return_counts=True)
        predicted_label = np.argmax(counts)
        if predicted_label==labels_test[recording]:
            accuracy_time+=1/len(labels_test)

print("Accuracy is; "+str(accuracy_time*100))
        

#%%
Net.hist_classification_test(labels_train,labels_test,number_of_labels)
#%% Save the network state. (Not Working) InvalidArgumentError: Cannot convert a Tensor of dtype resource to a NumPy array.
# import jsonpickle
# import json
# filename='Results/On_Off/Arx_test.txt'
# #Can't save the mlp cos of TypeError: can't pickle _thread.RLock objects
# Net.mlp=[]
# Net.hist_mlp_classification_train=[]
# Net.hist_mlp_classification_test=[]

# jsonNet = jsonpickle.encode(Net)
# savedata={}
# savedata['Net']=jsonNet

# with open(filename, 'w') as outfile: 
#     json.dump(savedata, outfile)


#%% Print Surfaces
# Method to plot reconstructed and original surfaces
Net.plt_surfaces_vs_reconstructions(file=10, layer=0, test=True)

#%% Net history plot
# Method to print loss history of the network
Net.plt_loss_history(layer=0)

#%% Print last layer activity 
# Method to plot last layer activation of the network
Net.plt_last_layer_activation(file=6, labels=labels_train, labels_test=labels_test,
                              classes=classes, test=True)

     
#%% Reverse activation
# Method to plot reverse activation of a sublayer output (output related to input)
Net.plt_reverse_activation(file=5, layer=0, sublayer=0, labels=labels_train, 
                           labels_test=labels_test, classes=classes, test=True)

#%% Print Histogram 
file = 6
file_response=Net.last_layer_activity_test[file]
plt.figure()
plt.hist(file_response[1],bins=Net.features_number[0][1])

#%% Plot T centers
plt.figure()
plt.plot(np.transpose(Net.local_sublayer[1].cluster_centers_))


#%% Plot 2D centers
center_no = 3
plt.figure()
plt.title("Center  nunber: "+ str(center_no))
plt.imshow(Net.cross_sublayer[0].cluster_centers_[center_no].reshape([32,6]))
plt.imshow(Net.cross_sublayer[1].cluster_centers_[center_no].reshape([8,12]))


#%% t-SNE to look at the quality of the clustering.

from sklearn.manifold import TSNE

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):

    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=y)    
    if title is not None:
        plt.title(title)
        plt.colorbar()  


localsurf=Net.surfaces[0][0:-1:1000] #[all_local_surfaces, all_local_surfaces_test, all_surfaces, all_surfaces_test]
crosssurf=Net.surfaces[2][0:-1:1000]
local_labels = Net.sub_t[0].predict(localsurf)
cross_labels = Net.sub_2D[0].predict(crosssurf)

tsne = TSNE(n_components=2, init='pca', random_state=0)
local_embedd=tsne.fit_transform(localsurf)
cross_embedd=tsne.fit_transform(crosssurf)

plot_embedding(local_embedd, local_labels, "t-SNE 1D HOTS")
plot_embedding(cross_embedd, cross_labels, "t-SNE 2D HOTS")

#%% t-SNE to look at the evolution of surfaces. 
from sklearn.manifold import TSNE

n_events_by_recor=[len(dataset_train[recording][0]) for recording in range(len(dataset_train))]
recording=2
N_events_upto_rec=sum(n_events_by_recor[:recording])
N_events_in_rec=n_events_by_recor[recording]

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):

    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=y)    
    if title is not None:
        plt.title(title)
        plt.colorbar()
        
recording_relative_timestamps=dataset_train[recording][0]/1.0
localsurf_recording=Net.surfaces[0][N_events_upto_rec:N_events_upto_rec+N_events_in_rec] #[all_local_surfaces, all_local_surfaces_test, all_surfaces, all_surfaces_test]
crosssurf_recording=Net.surfaces[2][N_events_upto_rec:N_events_upto_rec+N_events_in_rec] #[all_local_surfaces, all_local_surfaces_test, all_surfaces, all_surfaces_test]

tsne = TSNE(n_components=2, init='pca', random_state=0)
local_embedd_recording=tsne.fit_transform(localsurf_recording)
cross_embedd_recording=tsne.fit_transform(crosssurf_recording)

plot_embedding(local_embedd, recording_relative_timestamps, "t-SNE 1D timecoded surfaces total")
plot_embedding(cross_embedd_recording, recording_relative_timestamps, "t-SNE 1D timecoded surfaces total")


l_index=0
up_index=500
plot_embedding(local_embedd[l_index:up_index], recording_relative_timestamps[l_index:up_index], "t-SNE 1D timecoded surfaces timewindow")
plot_embedding(cross_embedd_recording[l_index:up_index], recording_relative_timestamps[l_index:up_index], "t-SNE 1D timecoded surfaces timewindow")


#%% PCA test
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


localsurf=Net.tmp[0]
crosssurf=Net.tmp[2]

pca_surf=IncrementalPCA(batch_size=6800)
pca_surf.fit(crosssurf)
exp_var=pca_surf.explained_variance_ratio_
n_dims = [i  for i in range(len(exp_var)) if sum(exp_var[:i])>0.95][0]
embedd = pca_surf.transform(crosssurf)[:,:n_dims]

#%% PCA visual
from sklearn.decomposition import PCA
from matplotlib import offsetbox

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], color=plt.cm.Set1(y / (features_number[0][1])))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / (features_number[0][1])),
                 fontdict={'weight': 'bold', 'size': 20})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            center_no=centers[i]
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(Net.PCA.inverse_transform(Net.aenc_2D[0].cluster_centers_[center_no]).reshape([32,6]), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

localsurf=Net.tmp[0]
crosssurftest=Net.tmp[2][:10000]
pca_surf=PCA(n_components=80)
surf_embedd=pca_surf.fit_transform(crosssurftest)
centers = Net.aenc_2D[0].predict(surf_embedd)
# t_centers = hdbscan.approximate_predict(Net.aenc_T[0], localsurf) 
pca_surf=PCA(n_components=2)
embedd=pca_surf.fit_transform(crosssurftest)
plot_embedding(embedd, centers, "PCA 2D HOTS")
# pca_surf=PCA(n_components=2)
# embedd=pca_surf.fit_transform(localsurf)
# plot_embedding(embedd, t_centers, "PCA T HOTS")
