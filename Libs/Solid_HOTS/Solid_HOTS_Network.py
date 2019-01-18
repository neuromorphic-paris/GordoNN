 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:09:42 2019

@author: marcorax

This file contains the Solid_HOTS_Net class 
used to exctract features from serialised tipe of data as speech 
 
"""

import numpy as np 
import time
from scipy import optimize
from scipy.spatial import distance 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from Libs.Solid_HOTS.Context_Surface_generators import Time_context, Time_Surface


# Class for Solid_HOTS_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, live plotting, and classification of the input data
# =============================================================================
class Solid_HOTS_Net:
    # Network constructor settings, the net is set in a random state,
    # with random basis and activations to define a starting point for the 
    # optimization algorithms
    # =============================================================================
    #
    # =============================================================================            
    def __init__(self, basis_number, context_lengths, input_channels, taus_T, taus_2D, 
                 exploring=False, net_seed = 0):
        self.basis_T = []
        self.activations_T = []
        self.basis_2D = []
        self.activations_2D = []
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(basis_number)
        self.context_lengths = context_lengths
        self.basis_number = basis_number
        self.polarities = []
        self.exploring = exploring
        ## Set of attributes used only in exploring mode
        if exploring is True:
            # attribute containing all 2D surfaces computed during a run in each layer 
            self.surfaces = []
            # attribute containing all 1D contexts computed during a run in each layer
            self.contexts = []
            # attribute containing all optimization errors computed during a run in each layer 
            self.errors = []
       
        # setting the seed
        rng = np.random.RandomState()
        if (net_seed!=0):
            rng.seed(net_seed)
        
        
        # Random initial conditions 
        for layer in range(self.layers):
            self.basis_T.append(rng.rand(basis_number[layer][0], context_lengths[layer]))
            self.activations_T.append(rng.rand(1,basis_number[layer][0]))
            if layer == 0:
                self.polarities.append(input_channels)
            else:
                self.polarities.append(basis_number[layer][1])
            self.basis_2D.append(rng.rand(basis_number[layer][1], self.polarities[layer]*basis_number[layer][0]))
            self.activations_2D.append(rng.rand(1,basis_number[layer][1]))         
            
    #TODO Check it and comment it
    def learn(self, dataset):
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            context_num = 0
            for batch in range(len(layer_dataset)):
                context_num += len(layer_dataset[batch][0])       
            all_contexts = np.zeros([context_num, self.context_lengths[layer]],dtype=float) # Preallocating is FUN!
            all_surfaces = np.zeros([context_num, self.polarities[layer]*self.basis_number[layer][0]],dtype=float)*7. 
            
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' CONTEXTS GENERATION ---')
                start_time = time.time()
            pos = 0
            for batch in range(len(layer_dataset)):
                for ind in range(len(layer_dataset[batch][0])):
                    event_polarity = layer_dataset[batch][1][ind]
                    all_contexts[pos, :] = Time_context(ind, layer_dataset[batch],
                                                              self.taus_T[layer][event_polarity],
                                                              self.context_lengths[layer])
                    pos +=1
                gc.collect()
                if self.exploring is True:
                    print("\r","Contexts generation :", (batch+1)/len(layer_dataset)*100,"%", end="")
            if self.exploring is True:    
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' CONTEXTS CLUSTERING ---')
                start_time = time.time()
            # Training the features (the basis)
            kmeans = KMeans(n_clusters=self.basis_number[layer][0]).fit(all_contexts)
            self.basis_T[layer]=kmeans.cluster_centers_
            if self.exploring is True:
                print("\n Clustering took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            pos = 0
            net_T_response = []
            for batch in range(len(layer_dataset)):
                polarities = layer_dataset[batch][1]
                features = kmeans.labels_[pos:pos+len(layer_dataset[batch][0])]
                net_T_response.append([layer_dataset[batch][0], polarities, features])
                pos += len(layer_dataset[batch][0])
            
            # clearing some variables
            del kmeans, layer_dataset, all_contexts
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            pos = 0
            ydim,xdim = [self.polarities[layer], self.basis_number[layer][0]]
            for batch in range(len(net_T_response)):
                for ind in range(len(net_T_response[batch][0])):
                    all_surfaces[pos, :] = Time_Surface(xdim, ydim, ind,
                                self.taus_2D[layer], net_T_response[batch],
                                minv=0.1)
                    pos +=1
                gc.collect()
                if self.exploring is True:
                    print("\r","Contexts generation :", (batch+1)/len(net_T_response)*100,"%", end="")
            if self.exploring is True:
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES CLUSTERING ---')
            start_time = time.time()
            # Training the features (the basis)
            kmeans = KMeans(n_clusters=self.basis_number[layer][1]).fit(all_surfaces)
            self.self.basis_2D[layer] = kmeans.cluster_centers_
            if self.exploring is True:
                print("\n Clustering took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            pos = 0
            net_2D_response = []
            for batch in range(len(net_T_response)):
                features = kmeans.labels_[pos:pos+len(net_T_response[batch][0])]
                net_2D_response.append([net_T_response[batch][0], features])
                pos += len(net_T_response[batch][0])
            
            layer_dataset = net_2D_response
            # clearing some variables
            del kmeans, net_2D_response, all_surfaces
        self.last_layer_activity = layer_dataset
        
        
    def compute_response(self, dataset):
        layer_dataset = dataset
        for layer in range(self.layers):
            context_num = 0
            for batch in range(len(layer_dataset)):
                context_num += len(layer_dataset[batch][0])       
            all_contexts = np.zeros([context_num, self.context_lengths[layer]],dtype=float) # Preallocating is FUN!
            all_surfaces = np.zeros([context_num, self.polarities[layer]*self.basis_number[layer][0]],dtype=float) 
            
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' T RESPONSE COMPUTING ---')
                start_time = time.time()
            pos = 0
            for batch in range(len(layer_dataset)):
                for ind in range(len(layer_dataset[batch][0])):
                    event_polarity = layer_dataset[batch][1][ind]
                    all_contexts[pos, :] = Time_context(ind, layer_dataset[batch],
                                                              self.taus_T[layer][event_polarity],
                                                              self.context_lengths[layer])
                    pos +=1
                if self.exploring is True:
                    print("\r","Temporal sublayer response computing :", (batch+1)/len(layer_dataset)*100,"%", end="")
            if self.exploring is True:    
                print("Computing temporal sublayer response took %s seconds." % (time.time() - start_time))

            # Obtain Net activations
            pos = 0
            net_T_response = []
            for batch in range(len(layer_dataset)):
                polarities = layer_dataset[batch][1]
                features = kmeans.labels_[pos:pos+len(dataset[batch][0])]
                net_T_response.append([layer_dataset[batch][0], polarities, features])
                pos += len(layer_dataset[batch][0])
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            pos = 0
            ydim,xdim = [self.polarities[layer], self.basis_number[layer][0]]
            for batch in range(len(net_T_response)):
                for ind in range(len(net_T_response[batch][0])):
                    all_surfaces[pos, :] = Time_Surface(xdim, ydim, ind,
                                self.taus_2D[layer], net_T_response[batch],
                                minv=0.1)
                    pos +=1
                if self.exploring is True:
                    print("\r","Contexts generation :", (batch+1)/len(net_T_response)*100,"%", end="")
            if self.exploring is True:
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES CLUSTERING ---')
            start_time = time.time()
            # Training the features (the basis)
            kmeans = KMeans(n_clusters=self.self.basis_number[layer][1]).fit(all_surfaces)
            self.self.basis_2D[layer] = kmeans.cluster_centers_
            if self.exploring is True:
                print("\n Clustering took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            pos = 0
            net_2D_response = []
            for batch in range(len(layer_dataset)):
                features = kmeans.labels_[pos:pos+len(dataset[batch][0])]
                net_2D_response.append([layer_dataset[batch][0], features])
                pos += len(layer_dataset[batch][0])
            
            layer_dataset = net_2D_response

    def histogram_classification_train(self, labels, number_of_labels, number_of_adresses, dataset=0):
        if self.exploring is True:
            print('\n--- SIGNATURES COMPUTING ---')
            start_time = time.time()
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
        if self.exploring is True:
            print("\n Signature computing took %s seconds." % (time.time() - start_time))
        
    def histogram_classification_test(self, labels, number_of_labels, number_of_adresses, dataset=0):
        print('\n--- TEST HISTOGRAMS COMPUTING ---')
        start_time = time.time()
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
        print("\n Test histograms computing took %s seconds." % (time.time() - start_time))
        return prediction_rate, ch_prediction_rate#, ch_distances, ch_predicted_labels, ch_histograms, ch_norm_histograms, distances, predicted_labels, histograms, norm_histograms