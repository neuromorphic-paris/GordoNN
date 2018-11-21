import gc
import glob
import math
import random
import struct
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from AERDATAfile import *
from assign_closest_center import *
from extract_channel_activity import *
from get_filenames_dataset import *
from loadAERDATA import *
from time_context_generation import *
from operator import itemgetter



## Parameters
number_files_dataset = 100 #Number of files to use from the dataset for each class.
number_files_to_train_features = 20 #Number of files to train context.
n_channels = 32 #Number of channels of the sensor used. 

nb_clusters = 50
ratio_empty_ctx = 0.3 # below this ratio the context will be discarded #CURRENTLY NOT BEING USED


#Let's be deterministic
random.seed(0)



## Getting filenames from the dataset
print '\n--- GETTING FILENAMES FROM THE DATASET ---'
start_time = time.time()
[filenames_train, class_train, filenames_test, class_test] = get_filenames_dataset(number_files_dataset)
print("Getting filenames from the dataset took %s seconds." % (time.time() - start_time))
#print filenames_train
#print filenames_test



## Reading spikes from each of the files
print '\n--- READING SPIKES ---'
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


data_array = [[] for i in range(len(spikes_train))]  #Data array for Marco

for ind in range(n_channels*2): #For each address (that's why n_channels*2. Remember that each channel has two addresses: ON and OFF)

    ## Training feature extractor
    print '\n--- TRAINING FEATURE EXTRACTOR ---'
    start_time = time.time()
    spikes_feature_train_channelled = [0] * len(spikes_feature_train)    
    for file in range(len(spikes_feature_train)): #Extract timestamps of a specific address for each file in spikes_feature_train
        spikes_feature_train_channelled[file] = extract_channel_activity(spikes_feature_train[file], ind)

    tau = 200; # tau is an exponential decay, in microseconds
    context_size = 20

    all_train_context = time_context_generation(spikes_feature_train_channelled, tau, context_size)
    print("Training feature extractor took %s seconds." % (time.time() - start_time))



    ## Clustering
    print '\n--- CLUSTERING ---'
    start_time = time.time()
    #[~, centers.data] = kmeans(all_train_context, nb_clusters, 'Distance', 'cosine');
    kmeans = KMeans(n_clusters=nb_clusters).fit(all_train_context)
    centroids = kmeans.cluster_centers_
    print("Clustering took %s seconds." % (time.time() - start_time))



    ## Assign closest center
    print '\n--- ASSIGN CLOSEST CENTER ---'
    start_time = time.time()

    spikes_train_channelled = [0] * len(spikes_train)
    for file in range(len(spikes_train)): #Extract timestamps of a specific address for each file in spikes_train
        spikes_train_channelled[file] = extract_channel_activity(spikes_train[file], ind) 
    spikes_test_channelled = [0] * len(spikes_test)
    for file in range(len(spikes_test)): #Extract timestamps of a specific address for each file in spikes_test
        spikes_test_channelled[file] = extract_channel_activity(spikes_test[file], ind)

    metric_dist_assignement = 'cosine' #euclidean
    spikes_train_channelled_closest_centers = assign_closest_center(spikes_train_channelled, centroids, tau, context_size, metric_dist_assignement)
    spikes_test_channelled_closest_centers = assign_closest_center(spikes_test_channelled, centroids, tau, context_size, metric_dist_assignement)
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
    for file in range(len(spikes_train_channelled)):
        for spike in range(len(spikes_train_channelled[file])):
            array_sp = [class_train[file], spikes_train_channelled[file][spike], ind, spikes_train_channelled_closest_centers[file][spike]]
            data_array[file].append(array_sp)


##Sorting spikes based on the timestamp value for each of the files
for i in range(len(spikes_train)):
    data_array[i] = sorted(data_array[i], key=itemgetter(1))