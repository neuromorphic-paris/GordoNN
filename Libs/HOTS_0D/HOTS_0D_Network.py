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
        print("\n Clustering took %s seconds." % (time.time() - start_time))
        # Obtain Net activations
        pos = 0
        net_response = []
        for batch in range(len(dataset)):
            addresses = dataset[batch][1]
            features = kmeans.labels_[pos:pos+len(dataset[batch][0])]
            new_coordinates = np.array([features, addresses], dtype=int).transpose()
            polarity = np.zeros(len(dataset[batch][0]),dtype=int) # Adding polarity to deal with 2D HOTS
            net_response.append([dataset[batch][0], new_coordinates, polarity])
            pos += len(dataset[batch][0])
        self.net_response = net_response
            
    def compute_response(self, dataset):
        print('\n--- RESPONSE COMPUTING ---')
        start_time = time.time()
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
            addresses = dataset[batch][1]
            new_coordinates = np.array([features,addresses], dtype=int).transpose()
            polarity = np.zeros(len(dataset[batch][0]),dtype=int) # Adding polarity to deal with 2D HOTS
            net_response.append([dataset[batch][0], new_coordinates, polarity])
            print("\r","Response computing :", (batch+1)/len(dataset)*100,"%", end="")
        self.net_response = net_response
        print("\n Response computing took %s seconds." % (time.time() - start_time))

    def histogram_classification_train(self, labels, number_of_labels, number_of_adresses, dataset=0):
        if dataset != 0:
            self.compute_response(dataset)
        net_response=self.net_response
        # Normalization factors
        spikes_per_channel_and_label = np.zeros([number_of_adresses, number_of_labels])
        spikes_per_label = np.zeros([number_of_labels])
        ch_histograms = np.zeros([number_of_adresses, number_of_labels, self.feat_number])
        ch_norm_histograms = np.zeros([number_of_adresses, number_of_labels, self.feat_number])
        histograms = np.zeros([number_of_labels, self.feat_number])
        norm_histograms = np.zeros([number_of_labels, self.feat_number])
        for batch in range(len(net_response)):
            current_label = labels[batch]
            for ind in range(len(net_response[batch][0])):
                histograms[current_label, net_response[batch][1][ind,0]] += 1
                spikes_per_label[current_label] += 1
                ch_histograms[net_response[batch][1][ind,1], current_label, net_response[batch][1][ind,0]] +=1 
                spikes_per_channel_and_label[net_response[batch][1][ind,1], current_label] += 1
        # Compute the Normalised histograms
        for label in range(number_of_labels):
            norm_histograms[label,:] = histograms[label,:]/spikes_per_label[label]
            for address in range(number_of_adresses):
                ch_norm_histograms[address,label,:] = ch_histograms[address,label,:]/spikes_per_channel_and_label[address,label]
        self.histograms = histograms
        self.norm_histograms = norm_histograms
        self.ch_histograms = ch_histograms
        self.ch_norm_histograms = ch_norm_histograms                
    
    def histogram_classification_test(self, labels, number_of_labels, number_of_adresses, dataset=0):
        if dataset != 0:
            self.compute_response(dataset)
        net_response=self.net_response
        ch_histograms = np.zeros([number_of_adresses, len(net_response), self.feat_number])
        ch_norm_histograms = np.zeros([number_of_adresses, len(net_response), self.feat_number])
        histograms = np.zeros([len(net_response), self.feat_number])
        norm_histograms = np.zeros([len(net_response), self.feat_number])
        for batch in range(len(net_response)):
            # Normalization factor
            spikes_per_channel = np.zeros([number_of_adresses])
            for ind in range(len(net_response[batch][0])):
                histograms[batch, net_response[batch][1][ind,0]] += 1
                ch_histograms[net_response[batch][1][ind,1], batch, net_response[batch][1][ind,0]] +=1 
                spikes_per_channel[net_response[batch][1][ind,1]] += 1
            norm_histograms[batch,:] = histograms[batch,:]/len(net_response[batch][0])                
            for address in range(number_of_adresses):
                ch_norm_histograms[address, batch, :] = ch_histograms[address, batch, :]/spikes_per_channel[address]
        
        # compute the distances per each histogram from the models
        distances = []
        predicted_labels = []
        prediction_rate = np.zeros(3)
        ch_distances = []
        ch_predicted_labels = []
        ch_prediction_rate = np.zeros([number_of_adresses,3])
        for batch in range(len(dataset)):
            single_batch_distances = []
            for label in range(number_of_labels):
                single_label_distances = []  
                single_label_distances.append(distance.euclidean(histograms[batch], self.histograms[label]))
                single_label_distances.append(distance.euclidean(norm_histograms[batch], self.norm_histograms[label]))
                Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(norm_histograms[batch], self.norm_histograms[label])]) 
                single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
                single_batch_distances.append(single_label_distances)
            
            ch_single_batch_distances = []
            ch_single_batch_predicted_labels = []
            for address in range(number_of_adresses):
                single_batch_distances_per_address = []
                for label in range(number_of_labels):
                    single_label_distances = []  
                    single_label_distances.append(distance.euclidean(ch_histograms[address,batch], self.ch_histograms[address,label]))
                    single_label_distances.append(distance.euclidean(ch_norm_histograms[address,batch], self.ch_norm_histograms[address,label]))
                    Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(ch_norm_histograms[address,batch], self.ch_norm_histograms[address,label])])
                    single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
                    single_batch_distances_per_address.append(single_label_distances)
                single_batch_distances_per_address = np.array(single_batch_distances_per_address)
                ch_single_batch_distances.append(single_batch_distances_per_address)
                single_batch_predicted_labels_per_address = np.argmin(single_batch_distances_per_address, 0)
                ch_single_batch_predicted_labels.append(single_batch_predicted_labels_per_address)
                for dist in range(3):
                    if single_batch_predicted_labels_per_address[dist] == labels[batch]:
                        ch_prediction_rate[address, dist] += 1/len(dataset)

                
            single_batch_distances = np.array(single_batch_distances)
            single_batch_predicted_labels = np.argmin(single_batch_distances, 0)
            for dist in range(3):
                if single_batch_predicted_labels[dist] == labels[batch]:
                    prediction_rate[dist] += 1/len(dataset)
                
            distances.append(single_batch_distances)
            predicted_labels.append(single_batch_predicted_labels)
            ch_distances.append(ch_single_batch_distances)
            ch_predicted_labels.append(ch_single_batch_predicted_labels)
        
        return prediction_rate, ch_prediction_rate#, ch_distances, ch_predicted_labels, ch_histograms, ch_norm_histograms, distances, predicted_labels, histograms, norm_histograms
            