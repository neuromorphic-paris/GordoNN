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
import time

# Data loading Libraries
from Libs.Data_loading.AERDATA_file import AERDATA_file
from Libs.Data_loading.AERDATA_load import AERDATA_load
from Libs.Data_loading.get_filenames_dataset import get_filenames_on_off_dataset

# 0 dimensional HOTS
from Libs.HOTS_0D.HOTS_0D_Network import HOTS_0D_Net
from sklearn.cluster import KMeans


# Dataset Parameters
# =============================================================================
# number_files_dataset : the number of files to be loaded for each class (On, Off)
# train_test_ratio: ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
# shuffle_seed : The seed used to shuffle the data, if 0 it will be totally 
#                random (no seed used)
# use_all_addr : if False all off events will be dropped, and the total addresses
#                number will correspond to the number of channel of the cochlea
# =============================================================================

number_files_dataset = 60
train_test_ratio = 0.75
shuffle_seed = 12
use_all_addr = False

## Data Loading

print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
start_time = time.time()
[filenames_train, labels_train, filenames_test, labels_test] = get_filenames_on_off_dataset(number_files_dataset, train_test_ratio, shuffle_seed)
print("Getting filenames from the dataset took %s seconds." % (time.time() - start_time))

## Reading spikes from each of the files
print ('\n--- READING SPIKES ---')
start_time = time.time()

dataset_train = []
dataset_test = []

for train_file in range(len(filenames_train)):
    addresses, timestamps = AERDATA_load(filenames_train[train_file], use_all_addr)
    dataset_train.append([np.array(timestamps), np.array(addresses)])
for test_file in range(len(filenames_test)):
    addresses, timestamps = AERDATA_load(filenames_test[test_file], use_all_addr)
    dataset_test.append([np.array(timestamps), np.array(addresses)])

# Shuffle data
# Setting the random state for data shuffling
rng = np.random.RandomState()
if(shuffle_seed!=0):
    rng.seed(shuffle_seed+1)

# Shuffle the dataset and the labels with the same order
combined_data = list(zip(dataset_train, labels_train))
rng.shuffle(combined_data)
dataset_train[:], labels_train[:] = zip(*combined_data)

combined_data = list(zip(dataset_test, labels_test))
rng.shuffle(combined_data)
dataset_test[:], labels_test[:] = zip(*combined_data)

print("Reading spikes took %s seconds." % (time.time() - start_time))



#%% 0D Network setting and feature exctraction (aka basis learning)

# 0D Network settings
# =============================================================================
# feat_number: is the number of feature or centers used by the 0D network
# feat_size: is the length of the time context generated per each spike
# taus: is a list containing the time coefficient used for the time surface creations
#       for each channel
# =============================================================================

feat_number = 20 
feat_size = 8

# tau is an exponential decay, in microseconds
taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
        1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
         20126, 25227, 31621, 39636, 49682])
taucoeff = 0.5 # Moltiplication factor for all taus


Net_0D = HOTS_0D_Net(feat_number, feat_size, taucoeff*taus)

Net_0D.learn_offline(dataset_train)

dataset_train = Net_0D.net_response

#%% 
Net_0D.compute_response(dataset_test)

dataset_test = Net_0D.net_response


#%%

## Getting filenames from the dataset
print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
start_time = time.time()
[filenames_train, class_train, filenames_test, class_test] = get_filenames_dataset(number_files_dataset)
print("Getting filenames from the dataset took %s seconds." % (time.time() - start_time))
#print filenames_train
#print filenames_test



## Reading spikes from each of the files
print ('\n--- READING SPIKES ---')
start_time = time.time()
spikes_train = [0] * len(filenames_train)
spikes_test = [0] * len(filenames_test)

for train_file in range(len(filenames_train)):
    spikes_train[train_file] = loadAERDATA(filenames_train[train_file])
for test_file in range(len(filenames_test)):
    spikes_test[test_file] = loadAERDATA(filenames_test[test_file])

#print len(spikes_train)         #number of files
#print len(spikes_test)

print("Reading spikes took %s seconds." % (time.time() - start_time))


## Splitting train into feature train and classifier train

spikes_train, class_train = shuffle(spikes_train, class_train, random_state=0)

class_feature_train = class_train[0:number_files_to_train_features]
spikes_feature_train = spikes_train[0:number_files_to_train_features]
class_train = class_train[number_files_to_train_features:len(spikes_train)]
spikes_train = spikes_train[number_files_to_train_features:len(spikes_train)]


#data_array = [[] for i in range(len(spikes_train))]  #Data array for Marco
data_array_train = []
data_array_test = []

count = 0
all_train_context = []

for ind in range(n_channels): #For each address (that's why n_channels*2. Remember that each channel has two addresses: ON and OFF)
    
    ## Training feature extractor
    print ('\n--- TRAINING FEATURE EXTRACTOR ---')
    start_time = time.time()
    spikes_feature_train_channelled = [0] * len(spikes_feature_train)    
    for file in range(len(spikes_feature_train)): #Extract timestamps of a specific address for each file in spikes_feature_train
        spikes_feature_train_channelled[file] = extract_channel_activity(spikes_feature_train[file], ind*2)

    context_size = 20

    all_train_context.append(time_context_generation(spikes_feature_train_channelled, taus[ind]*taucoeff, context_size))
    print("Training feature extractor took %s seconds." % (time.time() - start_time))



## Clustering
print ('\n--- CLUSTERING ---')
start_time = time.time()
#[~, centers.data] = kmeans(all_train_context, nb_clusters, 'Distance', 'cosine');
kmeans = KMeans(n_clusters=nb_clusters).fit([j for i in all_train_context for j in i])
centroids = kmeans.cluster_centers_
print("Clustering took %s seconds." % (time.time() - start_time))


for ind in range(n_channels): #For each address (that's why n_channels*2. Remember that each channel has two addresses: ON and OFF)

    ## Assign closest center
    print ('\n--- ASSIGN CLOSEST CENTER ---')
    start_time = time.time()

    spikes_train_channelled = [0] * len(spikes_train)
    for file in range(len(spikes_train)): #Extract timestamps of a specific address for each file in spikes_train
        spikes_train_channelled[file] = extract_channel_activity(spikes_train[file], ind*2) 
    spikes_test_channelled = [0] * len(spikes_test)
    for file in range(len(spikes_test)): #Extract timestamps of a specific address for each file in spikes_test
        spikes_test_channelled[file] = extract_channel_activity(spikes_test[file], ind*2)

    metric_dist_assignement = 'cosine' #euclidean
    spikes_train_channelled_closest_centers = assign_closest_center(spikes_train_channelled, centroids, taus[ind]*taucoeff, context_size, metric_dist_assignement)
    spikes_test_channelled_closest_centers = assign_closest_center(spikes_test_channelled, centroids, taus[ind]*taucoeff, context_size, metric_dist_assignement)
    print("Assign closest center took %s seconds." % (time.time() - start_time))
    


    ##For merging this code with Marco's we don't need these parts of the code where the signatures are computed and the recognition is performed.
    """ ## Compute signatures of each file
    print '\n--- COMPUTE SIGNATURES OF EACH FILE ---'
    start_time = time.time()
    signs_train = np.zeros((len(spikes_train_channelled_closest_centers), nb_clusters))
    for i in range(len(spikes_train_channelled_closest_centers)):
        hist, bin_edges = np.histogram(spikes_train_channelled_closest_centers[i], nb_clusters)
        signs_train[i] = hist

    signs_test = np.zeros((len(spikes_test_channelled_closest_centers), nb_clusters))
    for i in range(len(spikes_test_channelled_closest_centers)):
        hist, bin_edges = np.histogram(spikes_test_channelled_closest_centers[i], nb_clusters)
        signs_test[i] = hist
    print("Compute signatures of each file took %s seconds." % (time.time() - start_time))
    


    ## Recognition task
    argmin = np.argmin(cdist(signs_train, signs_test), axis=0)
    print len(signs_train), len(signs_test)
    print argmin
    truth = class_test
    print truth
    pred = np.zeros(len(argmin))
    for i in range(len(argmin)):
        pred[i] = class_train[argmin[i]]
    print pred

    rate = 0
    for i in range(len(pred)):
        if(int(pred[i]) == int(truth[i])):
            rate +=1
    rate = np.float(rate) / len(truth)
    print rate """
    
    
    ## Preparing data for Marco    
    addr_array_train = []
    pol_array_train = []
    for f in range(len(spikes_train_channelled)):
        addr_array_train.append([ind for i in range(len(spikes_train_channelled[f]))])
        pol_array_train.append([0 for i in range(len(spikes_train_channelled[f]))])

    ## Preparing data for Marco    
    addr_array_test = []
    pol_array_test = []
    for f in range(len(spikes_test_channelled)):
        addr_array_test.append([ind for i in range(len(spikes_test_channelled[f]))])
        pol_array_test.append([0 for i in range(len(spikes_test_channelled[f]))])

    if count == 0:
        data_prep_ts_train = spikes_train_channelled
        data_prep_ts_test = spikes_test_channelled
        data_prep_centers_train = spikes_train_channelled_closest_centers
        data_prep_centers_test = spikes_test_channelled_closest_centers
        data_prep_addrs_train = addr_array_train
        data_prep_pol_train = pol_array_train
        data_prep_addrs_test = addr_array_test
        data_prep_pol_test = pol_array_test
        count +=1
    else:
        for file in range(len(spikes_train_channelled)):
            data_prep_ts_train[file] = np.concatenate((data_prep_ts_train[file], spikes_train_channelled[file]), axis=0)
            data_prep_centers_train[file] = np.concatenate((data_prep_centers_train[file], spikes_train_channelled_closest_centers[file]), axis=0)
            data_prep_addrs_train[file] = np.concatenate((data_prep_addrs_train[file], addr_array_train[file]), axis=0)
            data_prep_pol_train[file] = np.concatenate((data_prep_pol_train[file], pol_array_train[file]), axis=0)
        for file in range(len(spikes_test_channelled)):
            data_prep_ts_test[file] = np.concatenate((data_prep_ts_test[file], spikes_test_channelled[file]), axis=0)
            data_prep_centers_test[file] = np.concatenate((data_prep_centers_test[file], spikes_test_channelled_closest_centers[file]), axis=0)
            data_prep_addrs_test[file] = np.concatenate((data_prep_addrs_test[file], addr_array_test[file]), axis=0)
            data_prep_pol_test[file] = np.concatenate((data_prep_pol_test[file], pol_array_test[file]), axis=0)

##Sorting spikes based on the timestamp value for each of the files
for f in range(len(spikes_train)):
    sorted_indices = np.argsort(data_prep_ts_train[f])
    data_prep_ts_train[f] = data_prep_ts_train[f][sorted_indices]
    data_prep_centers_train[f] = data_prep_centers_train[f][sorted_indices]
    data_prep_addrs_train[f] = data_prep_addrs_train[f][sorted_indices]
    data_prep_pos = np.array([data_prep_centers_train[f],data_prep_addrs_train[f]]).transpose()
    data_prep_pol_train[f]=np.array(data_prep_pol_train[f])
    data_array_train.append([data_prep_ts_train[f], data_prep_pos, data_prep_pol_train[f]])

##Sorting spikes based on the timestamp value for each of the files
for f in range(len(spikes_test)):
    sorted_indices = np.argsort(data_prep_ts_test[f])
    data_prep_ts_test[f] = data_prep_ts_test[f][sorted_indices]
    data_prep_centers_test[f] = data_prep_centers_test[f][sorted_indices]
    data_prep_addrs_test[f] = data_prep_addrs_test[f][sorted_indices]
    data_prep_pos = np.array([data_prep_centers_test[f],data_prep_addrs_test[f]]).transpose()
    data_prep_pol_test[f]=np.array(data_prep_pol_test[f])
    data_array_test.append([data_prep_ts_test[f], data_prep_pos, data_prep_pol_test[f]])    
    
    
#%% Generate 2D net

# TODO unify naming and conventions to avoid renaming like this
# dataset renaming
dataset_learning = data_array_train
dataset_testing = data_array_test

# Plotting settings
plt.style.use("dark_background")

# 2D Network settings
# =============================================================================
# basis_number is a list containing the number of basis used for each layer
# basis_dimension: is a list containing the dimension of every base for each layer
# taus: is a list containing the time coefficient used for the time surface creations
#       for each layer, all three lists need to share the same lenght obv.
# first_layer_polarities : the number of distinct polarities in the input layer 
# shuffle_seed, net_seed : seed used for dataset shuffling and net generation,
#                       if set to 0 the process will be totally random
# delay_coeff : the coefficient used to linearly adress delay to outputs of each layer
#               out_timestamp = in_timestamp + delay_coeff*(1-abs(a_j))
# =============================================================================

basis_number = [30]
basis_dimension = [[nb_clusters,n_channels]] 
taus = [2000]
# The output of the first layer Hots is monopolar
first_layer_polarities = 1
shuffle_seed = 7
net_seed = 25

delay_coeff = 15000    
    
# Print an element to check if it's all right
file = 3

tsurface=Time_Surface_all(xdim=nb_clusters, ydim=n_channels, timestamp=dataset_learning[file][0][10], timecoeff=taus[0], dataset=dataset_learning[file], num_polarities=1, minv=0.1, verbose=False)
ax = sns.heatmap(tsurface, annot=False, cbar=False, vmin=0, vmax=1)
plt.show()

for i in range(10030,10170):
    print(i)
    tsurface=Time_Surface_all(xdim=nb_clusters, ydim=n_channels, timestamp=dataset_learning[file][0][i], timecoeff=taus[0], dataset=dataset_learning[file], num_polarities=1, minv=0.1, verbose=False)
    sns.heatmap(data=tsurface, ax=ax, annot=False, cbar=False, vmin=0, vmax=1)
    plt.draw()
    plt.pause(0.001)

# Generate the network
Net = HOTS_Sparse_Net(basis_number, basis_dimension, taus, first_layer_polarities, delay_coeff, net_seed)
    
#%% Learning-online-Exp distance and Thresh
print ('\n--- 2D HOTS feature extraction ---')
start_time = time.time()

sparsity_coeff = [0.5, 0.5, 2000000]
learning_rate = [1, 1, 6000]
noise_ratio = [1, 0, 50]
sensitivity = [0.1, 0.5, 12000]
channel = 9

Net.learn_online(dataset=dataset_learning,
                 channel = channel,
                 method="Exp distance", base_norm="Thresh",
                 noise_ratio=noise_ratio, sparsity_coeff=sparsity_coeff,
                 sensitivity=sensitivity,
                 learning_rate=learning_rate, verbose=False)

elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))

# Taking the steady state values to perform the other tests
sparsity_coeff = sparsity_coeff[1]
noise_ratio = noise_ratio[1]
sensitivity = sensitivity[1]

#%% Learning offline full batch

start_time = time.time()

sparsity_coeff = 0.8
learning_rate = 0.2        
max_steps = 5
base_norm_coeff = 0.0005
precision = 0.01
channel = 5

Net.learn_offline(dataset_learning, channel, sparsity_coeff, learning_rate, max_steps, base_norm_coeff, precision, verbose=False)
    
elapsed_time = time.time()-start_time
print("Learning elapsed time : "+str(elapsed_time))           
sensitivity = 0   
noise_ratio = 0 
#%% Plot Basis 

layer = 0
sublayer = 0
Net.plot_basis(layer, sublayer)
plt.show()    

#%% Classification train

#net_activity = Net.full_net_dataset_response(dataset_testing, channel, "Exp distance", 
#                                                      noise_ratio, 
#                                                      sparsity_coeff,
#                                                      sensitivity)

Net.histogram_classification_train(dataset_learning[0:40], channel,
                                   class_train[0:40], 
                                   2, "Exp distance", noise_ratio,
                                   sparsity_coeff, sensitivity)



#%% Classification test 

test_results = Net.histogram_classification_test(dataset_testing, channel,
                                                 class_test,
                                                 2, "Exp distance", noise_ratio,
                                                 sparsity_coeff, sensitivity) 
hist = np.transpose(Net.histograms)
norm_hist = np.transpose(Net.normalized_histograms)
test_hist = np.transpose(test_results[2])
test_norm_hist = np.transpose(test_results[3])

eucl = 0
norm_eucl = 0
bhatta = 0
for i,right_label in enumerate(class_test):
    eucl += (test_results[1][i][0] == right_label)/len(class_test)
    norm_eucl += (test_results[1][i][1] == right_label)/len(class_test)
    bhatta += (test_results[1][i][2] == right_label)/len(class_test)
