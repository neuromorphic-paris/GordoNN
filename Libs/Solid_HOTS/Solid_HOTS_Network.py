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
from scipy.spatial import distance 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import gc
from Libs.Solid_HOTS.Context_Surface_generators import Time_context, Time_Surface

#TODO change this as soon we will have the new timesurfaces
channel = 9

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
    # basis_number(list of int lists): the number of feature or centers used by the Solid network
    #                             the first index identifies the layer, the second one
    #                             is 0 for the centers of the 0D sublayer, and 1 for 
    #                             the 2D centers
    # context(list of int): the length of the time context generatef per each layer
    # input_channels(int): the total number of channels of the cochlea in the input files 
    # taus_T(list of float lists):  a list containing the time coefficient used for 
    #                              the context creations for each layer (first index)
    #                              and each channel (second index) 
    # taus_2D(list of float):  a list containing the time coefficients used for the 
    #                          creation of timesurfaces per each layer
    # exploring(boolean) : If True, the network will output messages to inform the 
    #                      the users about the current states and will save the 
    #                      basis at each update to build evolution plots (currently not 
    #                      available cos the learning is offline)
    # net_seed : seed used for net generation, if set to 0 the process will be totally random
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
                self.polarities.append(basis_number[layer-1][1])
            self.basis_2D.append(rng.rand(basis_number[layer][1], self.polarities[layer]*basis_number[layer][0]))
            self.activations_2D.append(rng.rand(1,basis_number[layer][1]))         
    
    # Network learning method, for now it based on kmeans and it is offline
    # =============================================================================  
    # dataset : the dataset used to learn the network features in a unsupervised 
    #           manner      
    # =============================================================================  
    def learn(self, dataset):
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            context_num = 0
            for batch in range(len(layer_dataset)):
                context_num += len(layer_dataset[batch][0])       
            all_contexts = np.zeros([context_num, self.context_lengths[layer]],dtype=float) # Preallocating is FUN!
            all_surfaces = np.zeros([context_num, self.polarities[layer]*self.basis_number[layer][0]],dtype=float)
            
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
            events_per_batch = [] # How many surfaces are generated per each batch
            timestamps_2D = []
            for batch in range(len(net_T_response)):
                number_of_e = 0
                timestamps_2D_batch = []
                for ind in range(len(net_T_response[batch][0])):
                    #TODO change this as soon we will have the new timesurfaces
                    # If the reference event is not from the selected channel ditch 
                    # the process and move to the next one
                    # This process is not needed in other layers as 
                    # the synchronization has already happened
                    if net_T_response[batch][1][ind] != channel & layer==0:
                        continue
                    all_surfaces[pos, :] = Time_Surface(xdim, ydim, ind,
                                self.taus_2D[layer], net_T_response[batch],
                                minv=0.1)
                    timestamps_2D_batch.append(net_T_response[batch][0][ind])
                    pos +=1
                    number_of_e += 1
                timestamps_2D.append(timestamps_2D_batch)
                events_per_batch.append(number_of_e)
                gc.collect()
                if self.exploring is True:
                    print("\r","Contexts generation :", (batch+1)/len(net_T_response)*100,"%", end="")
            
            #TODO change this as soon we will have the new timesurfaces
            if layer==0:
                all_surfaces= all_surfaces[:pos,:]

            if self.exploring is True:
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES CLUSTERING ---')
            start_time = time.time()
            # Training the features (the basis)
            kmeans = KMeans(n_clusters=self.basis_number[layer][1]).fit(all_surfaces)
            self.basis_2D[layer] = kmeans.cluster_centers_
            if self.exploring is True:
                print("\n Clustering took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            pos = 0
            net_2D_response = []
            for batch in range(len(net_T_response)):
                features = kmeans.labels_[pos:pos+events_per_batch[batch]]
                net_2D_response.append([np.array(timestamps_2D[batch]), features])
                pos += events_per_batch[batch]
            
            layer_dataset = net_2D_response
            # clearing some variables
            del kmeans, net_2D_response, all_surfaces
        
        self.last_layer_activity = layer_dataset
        
    # Method used to compute the full network response to a dataset
    # =============================================================================  
    # dataset : the input dataset for the network
    #
    # The last layer output is stored in .last_layer_activity
    # =============================================================================         
    def compute_response(self, dataset):
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            net_T_response = []
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' 0D COMPUTATION ---')
                start_time = time.time()
            pos = 0
            for batch in range(len(layer_dataset)):
                features=np.zeros(len(layer_dataset[batch][0]),dtype=int)
                for ind in range(len(layer_dataset[batch][0])):
                    event_polarity = layer_dataset[batch][1][ind]
                    context = Time_context(ind, layer_dataset[batch],
                                                              self.taus_T[layer][event_polarity],
                                                              self.context_lengths[layer])
                    dist = np.sum((self.basis_T[layer]-context)**2,axis=1)
                    features[ind] = np.argmin(dist)
                    pos +=1
                polarities = layer_dataset[batch][1]
                net_T_response.append([layer_dataset[batch][0], polarities, features])
                gc.collect()
                if self.exploring is True:
                    print("\r","Response computation :", (batch+1)/len(layer_dataset)*100,"%", end="")
            if self.exploring is True:    
                print("0D response computation took %s seconds." % (time.time() - start_time))
            
            # clearing some variables
            del layer_dataset
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' 2D COMPUTATION ---')
            start_time = time.time()
            pos = 0
            ydim,xdim = [self.polarities[layer], self.basis_number[layer][0]]
            net_2D_response = []
            events_per_batch = [] # How many surfaces are generated per each batch
            timestamps_2D = []
            for batch in range(len(net_T_response)):
                number_of_e = 0
                timestamps_2D_batch = []
                features = []
                for ind in range(len(net_T_response[batch][0])):
                    #TODO change this as soon we will have the new timesurfaces
                    # If the reference event is not from the selected channel ditch 
                    # the process and move to the next one
                    # This process is not needed in other layers as 
                    # the synchronization has already happened
                    if net_T_response[batch][1][ind] != channel & layer==0:
                        continue
                    surface = Time_Surface(xdim, ydim, ind,
                                self.taus_2D[layer], net_T_response[batch],
                                minv=0.1)
                    timestamps_2D_batch.append(net_T_response[batch][0][ind])
                    dist = np.sum((self.basis_2D[layer]-surface)**2,axis=1)
                    features.append(np.argmin(dist))
                    pos +=1
                timestamps_2D.append(timestamps_2D_batch)
                events_per_batch.append(number_of_e)
                net_2D_response.append([np.array(timestamps_2D[batch]), np.array(features)])
                gc.collect()
                if self.exploring is True:
                    print("\r","Response computation :", (batch+1)/len(net_T_response)*100,"%", end="")
            if self.exploring is True:
                print("2D response computation took %s seconds." % (time.time() - start_time))
            
            layer_dataset = net_2D_response
            # clearing some variables
            del net_2D_response
        
        self.last_layer_activity = layer_dataset

    # Method used to train the network histograms or signatures per each channel
    # =============================================================================  
    # labels(list of int) : a list containing the ordered labels for each batch
    # number_of_labels(int) : the total number of labels of the dataset
    # dataset : the input dataset for the network(if set to 0 the function will
    #           use the last dataset response computed if available)
    #
    # The last layer output is stored in .last_layer_activity
    # =============================================================================       
    def histogram_classification_train(self, labels, number_of_labels, dataset=0):
        if self.exploring is True:
            print('\n--- SIGNATURES COMPUTING ---')
            start_time = time.time()
        if dataset != 0:
            self.compute_response(dataset)
        net_response=self.last_layer_activity
        # Normalization factors
        spikes_per_label = np.zeros([number_of_labels])
        histograms = np.zeros([number_of_labels, self.basis_number[-1][1]])
        norm_histograms = np.zeros([number_of_labels, self.basis_number[-1][1]])
        for batch in range(len(net_response)):
            current_label = labels[batch]
            for ind in range(len(net_response[batch][0])):
                histograms[current_label, net_response[batch][1][ind]] += 1
                spikes_per_label[current_label] += 1
        # Compute the Normalised histograms
        for label in range(number_of_labels):
            norm_histograms[label,:] = histograms[label,:]/spikes_per_label[label]
        self.histograms = histograms
        self.normalized_histograms = norm_histograms
        if self.exploring is True:
            print("\n Signature computing took %s seconds." % (time.time() - start_time))
    
    # Method used to train the network histograms or signatures per each channel
    # =============================================================================  
    # labels(list of int) : a list containing the ordered labels for each batch
    # number_of_labels(int) : the total number of labels of the dataset
    # dataset : the input dataset for the network(if set to 0 the function will
    #           use the last dataset response computed if available)
    #
    # The last layer output is stored in .last_layer_activity
    # =============================================================================     
    def histogram_classification_test(self, labels, number_of_labels, dataset=0):
        print('\n--- TEST HISTOGRAMS COMPUTING ---')
        start_time = time.time()
        if dataset != 0:
            self.compute_response(dataset)
        net_response=self.last_layer_activity
        histograms = np.zeros([len(net_response), self.basis_number[-1][1]])
        norm_histograms = np.zeros([len(net_response), self.basis_number[-1][1]])
        for batch in range(len(net_response)):
            # Normalization factor
            for ind in range(len(net_response[batch][0])):
                histograms[batch, net_response[batch][1][ind]] += 1
            norm_histograms[batch,:] = histograms[batch,:]/len(net_response[batch][0])                
        
        # compute the distances per each histogram from the models
        distances = []
        predicted_labels = []
        prediction_rate = np.zeros(3)
        for batch in range(len(net_response)):
            single_batch_distances = []
            for label in range(number_of_labels):
                single_label_distances = []  
                single_label_distances.append(distance.euclidean(histograms[batch], self.histograms[label]))
                single_label_distances.append(distance.euclidean(norm_histograms[batch], self.normalized_histograms[label]))
                Bhattacharyya_array = np.array([np.sqrt(a*b) for a,b in zip(norm_histograms[batch], self.normalized_histograms[label])]) 
                single_label_distances.append(-np.log(sum(Bhattacharyya_array)))
                single_batch_distances.append(single_label_distances)
            
                
            single_batch_distances = np.array(single_batch_distances)
            single_batch_predicted_labels = np.argmin(single_batch_distances, 0)
            for dist in range(3):
                if single_batch_predicted_labels[dist] == labels[batch]:
                    prediction_rate[dist] += 1/len(net_response)
                
            distances.append(single_batch_distances)
            predicted_labels.append(single_batch_predicted_labels)
        self.test_histograms = histograms
        self.test_normalized_histograms = norm_histograms    
        print("\n Test histograms computing took %s seconds." % (time.time() - start_time))
        return prediction_rate, distances, predicted_labels
    
    # Method for plotting the histograms of the network, either result of train 
    # or testing
    # =============================================================================
    # label_names : tuple containing the names of each label that will displayed
    #               in the legend
    # labels : list containing the labels of the test dataset used to generate
    #          the histograms, if empty, the function will plot the class histograms
    #          computed using .histogram_classification_train
    # =============================================================================
    def plot_histograms(self, label_names, labels=[]):
        if labels == []:
            hist = np.transpose(self.histograms)
            norm_hist = np.transpose(self.normalized_histograms)
            eucl_fig, eucl_ax = plt.subplots()
            eucl_ax.set_title("Train histogram based on euclidean distance")
            eucl_ax.plot(hist)
            eucl_ax.legend(label_names)

            norm_fig, norm_ax = plt.subplots()
            norm_ax.set_title("Train histogram based on normalized euclidean distance")
            norm_ax.plot(norm_hist)
            norm_ax.legend(label_names)
        else:
            eucl_fig, eucl_ax = plt.subplots()
            eucl_ax.set_title("Test histogram based on euclidean distance")
            
            norm_fig, norm_ax = plt.subplots()
            norm_ax.set_title("Test histogram based on normalized euclidean distance")
            custom_lines = [Line2D([0], [0], color="C"+str(label), lw=1) for label in range(len(label_names))]
            for batch in range(len(labels)):
                eucl_ax.plot(self.test_histograms[batch].transpose(),"C"+str(labels[batch]))
                norm_ax.plot(self.test_normalized_histograms[batch].transpose(),"C"+str(labels[batch]))
            
            eucl_ax.legend(custom_lines,label_names)
            norm_ax.legend(custom_lines,label_names)

    
    # Method for plotting the basis set of a single sublayer
    # =============================================================================
    # layer(int) : the index of the selected layer 
    # sublayer(int) : the index of the selected sublayer (either 0 or 1)
    # =============================================================================
    def plot_basis(self, layer, sublayer):
        for i in range(self.basis_number[layer][sublayer]):
            if sublayer==1:
                plt.figure("Context prototype N: "+str(i)+" layer: "+str(layer))
                sns.heatmap(self.basis_2D[layer][i].reshape(self.polarities[layer],self.basis_number[layer][0]))
            if sublayer==0:
                plt.figure("Time surface prototype N: "+str(i)+" layer: "+str(layer))
                sns.heatmap([self.basis_T[layer][i]])