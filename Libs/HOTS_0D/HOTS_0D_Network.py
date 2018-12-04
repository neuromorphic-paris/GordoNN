#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:51:42 2018

@author: pedro, marcorax

This file contains a class used to create HOTS_0D networks, the related paper 
is still waiting to be published, the link will be added here as soon as possible.
As the paper is not yet out the convention for naming is the one adopted by the 
Sparse HOTS paper:

https://arxiv.org/abs/1804.09236

 
"""
import numpy as np 
import time
from scipy import optimize
from scipy.spatial import distance 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from Libs.HOTS_0D.Time_context_generators import Time_context

# Class for HOTS_0D_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, reconstruction and classification of the input data
# =============================================================================
class HOTS_0D_Net:
    
    def __init__(self, feat_number, feat_size, taus):
        self.feat_number = feat_number
        self.feat_size = feat_size
        self.taus = taus
        
    def learn_offline(self, dataset):
        all_contexts_length = 0
        for batch in range(len(dataset)):
            all_contexts_length += len(dataset[batch][0])       
        all_contexts = np.zeros([all_contexts_length, self.feat_size],dtype=float) # Preallocating is FUN!
        print('\n--- CONTEXTS GENERATION ---')
        start_time = time.time()
        pos = 0
        for batch in range(len(dataset)):
            for ind in range(len(dataset[batch][0])):
                event = [dataset[batch][0][ind], dataset[batch][1][ind]]
                channel = event[1]
                all_contexts[pos, :] = Time_context(ind, dataset[batch],
                                                          self.taus[channel],
                                                          self.feat_size)
                pos +=1
            print("\r","Contexts generation :", (batch+1)/len(dataset)*100,"%", end="")
        all_contexts
        print("Generating contexts took %s seconds." % (time.time() - start_time))
        print('\n--- CLUSTERING ---')
        start_time = time.time()
        # Training the features (the basis)
        kmeans = KMeans(n_clusters=self.feat_number).fit(all_contexts)
        self.basis = kmeans.cluster_centers_
        print("Clustering took %s seconds." % (time.time() - start_time))
        # Obtain Net activations
        pos = 0
        net_response = []
        for batch in range(len(dataset)):
            addresses = dataset[batch][0]
            features = kmeans.labels_[pos:pos+len(dataset[batch][0])]
            new_coordinates = np.array([addresses,features]).transpose
            polarity = np.zeros(len(dataset[batch][0])) # Adding polarity to deal with 2D HOTS
            net_response.append([dataset[batch][0], new_coordinates, polarity])
            pos += len(dataset[batch][0])
        self.net_response = net_response
            
    def compute_response(self, dataset):        
        net_response = []
        for batch in range(len(dataset)):
            features=np.zeros(len(dataset[batch][0]))
            for ind in range(len(dataset[batch][0])):
                event = [dataset[batch][0][ind], dataset[batch][1][ind]]
                channel = event[1]
                context = Time_context(ind, dataset[batch], self.taus[channel],
                                                              self.feat_size)
                
                dist = np.sum((self.basis-context)**2,axis=1)
                features[ind] = np.argmin(dist)
            addresses = dataset[batch][0]
            new_coordinates = np.array([addresses,features]).transpose
            polarity = np.zeros(len(dataset[batch][0])) # Adding polarity to deal with 2D HOTS
            net_response.append([dataset[batch][0], new_coordinates, polarity])
            print("\r","Response computing :", (batch+1)/len(dataset)*100,"%", end="")
        self.net_response = net_response

        