#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:09:42 2019

@author: marcorax

This file contains the Solid_HOTS_Net class 
used to exctract features from serialised tipe of data as speech 
 
"""
# General purpouse libraries
import time
import gc
import numpy as np 
from joblib import Parallel, delayed 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan

# Homemade Fresh Libraries like Grandma does
from Libs.Solid_HOTS._General_Func import create_autoencoder, events_from_activations,\
     local_surface_plot, surfaces_plot,\
     recording_local_surface_generator, recording_local_surface_generator_rate,\
     recording_surface_generator, compute_abs_ind
     



# Class for Solid_HOTS_Net
# =============================================================================
# This class contains the network data structure as the methods for performing
# learning, live plotting, and classification of the input data
# =============================================================================
class Solid_HOTS_Net:
    """
    Network constructor settings, the net is set in a random state,
    with random basis and activations to define a starting point for the 
    optimization algorithms
    Arguments: 
    features_number (nested lists of int) : the number of feature or centers used by the Solid network,
                                the first index identifies the layer, the second one
                                is the number of  for units of the 0D sublayer,
                                and the third for the 2D units
    l1_norm_coeff (nested lists of int) : Same structure of feature_number but used to store l1 normalization to sparify 
                                          bottleneck layer of each autoencoder
    learning_rate (nested lists of int) : Same structure of feature_number but 
                                          used to store the learning rates of 
                                          each autoencoder
    epochs (nested lists of int) : Same structure of feature_number but 
                                          used to store the epochs to train 
                                          each autoencoder
    local_surface_length (list of int): the length of the Local time surfaces generated per each layer
    input_channels (int) : thex total number of channels of the cochlea in the input files 
    taus_T(list of float lists) :  a list containing the time coefficient used for 
                                   the local_surface creations for each layer (first index)
                                   and each channel (second index). To keep it simple, 
                                   it's the result of a multiplication between a vector for each 
                                   layer(channel_taus) and a coefficient (taus_T_coeff).
    taus_2D (list of float) : a list containing the time coefficients used for the 
                             creation of timesurfaces per each layer
    batch_size (list of int) : a list containing the batch sizes used for the 
                             training of each layer
    activity_th (float) : The code will check that the sum(local surface)
    intermediate_dim_T (int) : Number of units used for intermediate layers
    intermediate_dim_2D (int) : Number of units used for intermediate layers
    threads (int) : The network can compute timesurfaces in a parallel way,
                    this parameter set the number of multiple threads allowed to run
 

    exploring (boolean) : If True, the network will output messages to inform the 
                          the users about the current states as well as plots
    """
    def __init__(self, network_parameters):
        
        [[features_number, l1_norm_coeff, learning_rate, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, exploring], [learning_rate,\
        epochs, l1_norm_coeff, intermediate_dim_T, intermediate_dim_2D,\
        activity_th, batch_size, spacing_local_T]] = network_parameters
        
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        self.input_channels = input_channels
        
        self.batch_size=batch_size
        self.intermediate_dim_T=intermediate_dim_T
        self.intermediate_dim_2D=intermediate_dim_2D
        self.activity_th = activity_th
        self.spacing_local_T = spacing_local_T
        self.learning_rate = learning_rate
        self.l1_norm_coeff = l1_norm_coeff
        self.epochs = epochs
        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.exploring = exploring
      
    
    # =============================================================================  
    def learn(self, dataset, dataset_test, labels, labels_test, num_classes, rerun_layer=0):
        """
        Network learning method, it also processes the test dataset, to show 
        reconstruction error on it too. It saves net responses recallable with 
        method .last_layer_activity and .last_layer_activity_test
        They are used subsequently to train and test the classifierss
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner
            dataset_test (float) : It's the test dataset used to produce the test surfaces        
            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        # Create a copy of the dataset, the network might remove data without 
        # enough and the original data might be affected
        layer_dataset = dataset.copy()
        layer_dataset_test = dataset_test.copy()

        #check if you only want to rerun a layer
        if rerun_layer == 0:
            self.aenc_T=[] # Autoencoder models for 0D Sublayers
            self.aenc_2D=[] # Autoencoder models for 2D Sublayers
            layers_index = np.arange(self.layers)
            self.tmporig=[]
            self.tmpcr=[]
            self.tmporig_c=[]
            self.tmpcr_c=[]
            self.net_0D_response=[]
            self.net_0D_response_test=[]
            self.net_2D_response = []
            self.net_2D_response_test = []
            self.layer_dataset = []
            self.layer_dataset_test = []

            # The code is going to run on gpus, to improve performances rather than 
            # a pure online algorithm I am going to minibatch 
            
            self.removed_ind = []
            self.removed_ind_test = []
        else:
            layers_index = np.arange(rerun_layer,self.layers)

                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset.append(layer_dataset)
                self.layer_dataset_test.append(layer_dataset_test)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset=self.net_2D_response[layer-1]
                    layer_dataset_test=self.net_2D_response_test[layer-1]
                    self.layer_dataset=self.layer_dataset[:layer+1]
                    self.layer_dataset_test=self.layer_dataset_test[:layer+1]
              
                self.aenc_T=self.aenc_T[:layer]
                self.aenc_2D=self.aenc_2D[:layer]
                self.tmporig_c = self.tmporig_c[:layer]
                self.tmpcr_c = self.tmpcr_c[:layer]
                self.net_0D_response = self.net_0D_response[:layer]
                self.net_0D_response_test = self.net_0D_response_test[:layer]
                self.net_2D_response = self.net_2D_response[:layer]
                self.net_2D_response_test = self.net_2D_response_test[:layer]
                self.tmporig = self.tmporig[:layer]
                self.tmpcr = self.tmpcr[:layer]
                self.removed_ind = self.removed_ind[:layer]
                self.removed_ind_test = self.removed_ind_test[:layer]
                
            
                
            # Create the autoencoder for this layer
            
            # self.aenc_T.append(create_autoencoder(self.local_surface_length[layer],
            #                               self.features_number[layer][0],
            #                               self.intermediate_dim_T,
            #                               self.learning_rate[layer][0],
            #                               self.l1_norm_coeff[layer][0],
            #                               self.exploring))

            # GENERATING AND COMPUTING TIME LOCAL SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' LOCAL SURFACES GENERATION ---')
                start_time = time.time()
           
            

                
            # Lists used to store removed files indices due to lack of data
            # It might happen when .activity_th is too high 
            removed_ind_layer = []
            removed_ind_layer_test = []
            
            # Temporary lists used to store all the local surfaces produced here
            all_local_surfaces=[]
            local_surface_indices=[]
            all_local_surfaces_test=[]
            local_surface_indices_test=[]
            
            total_events=0                                  
            
            # The first layer doesn't have rate, thus i have to check that
            if layer == 0 :
                # Generation of local surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads)(delayed(recording_local_surface_generator)\
                                    (recording, layer_dataset, self.polarities[layer],\
                                    self.taus_T[layer], self.local_surface_length[layer],
                                    self.exploring, self.activity_th, self.spacing_local_T[layer])\
                                    for recording in range(len(layer_dataset)))
            else:
                # Generation of local surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads)(delayed(recording_local_surface_generator_rate)\
                                    (recording, layer_dataset, self.polarities[layer],\
                                    self.taus_T[layer], self.local_surface_length[layer],
                                    self.exploring, self.activity_th, self.spacing_local_T[layer])\
                                    for recording in range(len(layer_dataset)))        
             
            for recording in range(len(layer_dataset)):                    
                all_local_surfaces.append(results[recording][1])
                local_surface_indices.append(results[recording][0])
                total_events+=len(local_surface_indices[-1])
            
            
            # some files might not have enough events after the first layer, 
            # i remove them and save the dataset indices, i will then find 
            # the absolute indices to know which files were discarded                
            lost_ind = [i for i in range(len(all_local_surfaces)) if all_local_surfaces[i].size == 0]
 
            # i need to go backwards or the indices will change after first delition
            for i in lost_ind[-1::-1]:
                del all_local_surfaces[i]
                del local_surface_indices[i]
                del layer_dataset[i]
            removed_ind_layer=lost_ind
            
            # The final results of the local surfaces train dataset computation
            all_local_surfaces=np.concatenate(all_local_surfaces, axis=0)
            all_local_surfaces = np.array(all_local_surfaces, dtype='float16')
            
            # Test dataset
            total_events=0

            # The first layer doesn't have rate, thus i have to check that
            if layer == 0 :
                # Generation of local surfaces, computed on multiple threads                   
                results = Parallel(n_jobs=self.threads)(delayed(recording_local_surface_generator)\
                                    (recording, layer_dataset_test, self.polarities[layer],\
                                    self.taus_T[layer], self.local_surface_length[layer],
                                    self.exploring, self.activity_th, self.spacing_local_T[layer])\
                                    for recording in range(len(layer_dataset_test)))
            else:
                # Generation of local surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads)(delayed(recording_local_surface_generator_rate)\
                                    (recording, layer_dataset_test, self.polarities[layer],\
                                    self.taus_T[layer], self.local_surface_length[layer],
                                    self.exploring, self.activity_th, self.spacing_local_T[layer])\
                                    for recording in range(len(layer_dataset_test)))   
           
            gc.collect()
            
            # unpacking the results                   
            for recording in range(len(layer_dataset_test)):                    
                all_local_surfaces_test.append(results[recording][1])
                local_surface_indices_test.append(results[recording][0])
                total_events+=len(local_surface_indices_test[-1])


            # some files might not have enough events after the first layer, 
            # i remove them and save the dataset indices, i will then find 
            # the absolute indices to know which files were discarded         
            lost_ind = [i for i in range(len(all_local_surfaces_test)) if all_local_surfaces_test[i].size == 0]
            # i need to go backwards or the indices will change after first delition
            for i in lost_ind[-1::-1]:
                del all_local_surfaces_test[i]
                del local_surface_indices_test[i]
                del layer_dataset_test[i]
            removed_ind_layer_test=lost_ind
            
            # The final results of the local surfaces test dataset computation
            all_local_surfaces_test = np.concatenate(all_local_surfaces_test, axis=0)
            all_local_surfaces_test = np.array(all_local_surfaces_test, dtype='float16')


            # TRAINING LOCAL SURFACES FEATURES
            if self.exploring is True:    
                print("Generating local_surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' LOCAL SURFACES FEATURES EXTRACTION ---')
                start_time = time.time()
                
            self.aenc_T.append(KMeans(n_clusters=self.features_number[layer][0], random_state=0).fit(all_local_surfaces))

            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
                local_surface_plot(layer_dataset, local_surface_indices, all_local_surfaces, layer)
                local_surface_plot(layer_dataset_test, local_surface_indices_test, all_local_surfaces_test, layer)
                
                
            # Compute 0D sublayer activations
            # Train set
            current_pos = 0
            net_0D_response = []
            for recording in range(len(layer_dataset)):                
                # Get network activations at steady state (after learning)
                recording_results = self.aenc_T[layer].predict(np.array(all_local_surfaces[current_pos:current_pos+len(local_surface_indices[recording])]))                                
                current_pos += len(local_surface_indices[recording])
                net_0D_response.append(events_from_activations(recording_results, local_surface_indices[recording], layer_dataset[recording]))
            # Test set            
            current_pos = 0
            net_0D_response_test = []
            for recording in range(len(layer_dataset_test)):                
                # Get network activations at steady state (after learning)
                recording_results = self.aenc_T[layer].predict(np.array(all_local_surfaces_test[current_pos:current_pos+len(local_surface_indices_test[recording])]))
                current_pos += len(local_surface_indices_test[recording])
                net_0D_response_test.append(events_from_activations(recording_results, local_surface_indices_test[recording], layer_dataset_test[recording]))
            
            
            # clearing some variables
            gc.collect()
            
            # Save the 0D response
            self.net_0D_response.append(net_0D_response)
            self.net_0D_response_test.append(net_0D_response_test)
            
            # Create the varational autoencoder for this layer
            # self.aenc_2D.append(create_autoencoder(self.polarities[layer]*self.features_number[layer][0],
            #                             self.features_number[layer][1], self.intermediate_dim_2D, self.learning_rate[layer][1],  self.l1_norm_coeff[layer][1], self.exploring))
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            
            # Temporary lists used to store all the local surfaces produced here one for each class
            all_surfaces=[] 
            surface_indices=[]
            all_surfaces_test=[]
            surface_indices_test=[]
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads)(delayed(recording_surface_generator)\
                                                                  (recording,
                                                                   net_0D_response, 
                                                                   self.polarities[layer],
                                                                   self.features_number[layer],
                                                                   self.taus_2D[layer],
                                                                   self.exploring)\
                                    for recording in range(len(net_0D_response)))
           
            # unpacking the results                   
            for recording in range(len(net_0D_response)):                    
                all_surfaces.append(results[recording][1])
                surface_indices.append(results[recording][0])
           
            all_surfaces = np.concatenate(all_surfaces, axis=0) 

            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads)(delayed(recording_surface_generator)\
                                                                  (recording,
                                                                   net_0D_response_test, 
                                                                   self.polarities[layer],
                                                                   self.features_number[layer],
                                                                   self.taus_2D[layer],
                                                                   self.exploring)\
                                    for recording in range(len(net_0D_response_test)))
           
            # unpacking the results                   
            for recording in range(len(net_0D_response_test)):                    
                all_surfaces_test.append(results[recording][1])
                surface_indices_test.append(results[recording][0])
                
            all_surfaces_test = np.concatenate(all_surfaces_test, axis=0) 
            
            if self.exploring is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            
            # The final results of the local surfaces test dataset computation
            all_surfaces = np.array(all_surfaces, dtype='float16') 
            all_surfaces_test = np.array(all_surfaces_test, dtype='float16')
            gc.collect()
            
            
            # TRAINING CROSS SURFACES FEATURES
            print('PCA')
            pca_surf=PCA(n_components=3)
            surf_embedd=pca_surf.fit_transform(all_surfaces)
            print('PCA_DONE')
            # self.aenc_2D.append(KMeans(n_clusters=self.features_number[layer][1], verbose=1).fit(all_surfaces))
            self.aenc_2D.append(hdbscan.HDBSCAN(min_cluster_size=300, core_dist_n_jobs=24, prediction_data=True))
            self.aenc_2D[layer].fit(surf_embedd)
            self.features_number[layer][-1]=np.max(self.aenc_2D[layer].labels_)+1
            
            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
                surfaces_plot(all_surfaces, self.polarities[layer], self.features_number[layer][0])
                surfaces_plot(all_surfaces_test, self.polarities[layer], self.features_number[layer][0])

            current_pos = 0
            current_pos_test = 0
            

            # Compute 0D sublayer activations
            
            layer_2D_activations=[]
            layer_2D_activations_test=[]
            
            # Train set            
            for recording in range(len(surface_indices)):
                surf_embedd=pca_surf.fit_transform(np.array(all_surfaces[current_pos:current_pos+len(surface_indices[recording])]))
                recording_results, strength = hdbscan.approximate_predict(self.aenc_2D[layer], surf_embedd)
                # recording_results = self.aenc_2D[layer].predict(np.array(all_surfaces[current_pos:current_pos+len(surface_indices[recording])]))
                current_pos += len(surface_indices[recording])
                layer_2D_activations.append([net_0D_response[recording][0][surface_indices[recording]], recording_results])
            del net_0D_response

            # Test set            
            for recording in range(len(surface_indices_test)):
                surf_embedd=pca_surf.fit_transform(np.array(all_surfaces_test[current_pos_test:current_pos_test+len(surface_indices_test[recording])]))
                recording_results, strength = hdbscan.approximate_predict(self.aenc_2D[layer], surf_embedd)
                # recording_results = self.aenc_2D[layer].predict(np.array(all_surfaces_test[current_pos_test:current_pos_test+len(surface_indices_test[recording])]))
                current_pos_test += len(surface_indices_test[recording])
                layer_2D_activations_test.append([net_0D_response_test[recording][0][surface_indices_test[recording]], recording_results])
            del net_0D_response_test
            self.tmp=[all_local_surfaces, all_local_surfaces_test, all_surfaces, all_surfaces_test]
            self.net_2D_response.append(layer_2D_activations)
            self.net_2D_response_test.append(layer_2D_activations_test)
            
            layer_dataset = layer_2D_activations
            layer_dataset_test = layer_2D_activations_test

            self.removed_ind.append(removed_ind_layer)
            self.removed_ind_test.append(removed_ind_layer_test)
                        
            # clearing some rubbish
            gc.collect()
            
  
        self.last_layer_activity = layer_2D_activations
        self.last_layer_activity_test = layer_2D_activations_test
       
        # compute absolute removed indices      
        self.abs_rem_ind = compute_abs_ind(self.removed_ind)
        self.abs_rem_ind_test = compute_abs_ind(self.removed_ind_test)


    def add_layers(self, network_parameters):        
        """
        # If you want to add a layer or more on top of the net and keep the results
        # you had for the first ones use this. 
        # You will have to load new parameters and run the learning from the 
        # index of the new layer (You computed a 2 layer network, you want to add 2 more,
        # you use a new set of parameters, and rerun learning with rerun_layer=2, as the 
        # layers index start with 0.
        Net.add_layers(network_parameters)
        Net.learn(dataset_train, dataset_test, rerun_layer = 2)
        """
        [[features_number, l1_norm_coeff, learning_rate, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, exploring], [learning_rate,\
        epochs, l1_norm_coeff, intermediate_dim_T, intermediate_dim_2D,\
        activity_th, batch_size, spacing_local_T]] = network_parameters
        
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        
        self.batch_size=batch_size
        self.intermediate_dim_T=intermediate_dim_T
        self.intermediate_dim_2D=intermediate_dim_2D
        self.activity_th = activity_th
        self.spacing_local_T = spacing_local_T
        self.learning_rate = learning_rate
        self.l1_norm_coeff = l1_norm_coeff
        self.epochs = epochs
        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.exploring = exploring
        
        # Load the results of the last layer as input for the next one.
        self.layer_dataset.append(self.last_layer_activity)
        self.layer_dataset_test.append(self.last_layer_activity_test)
    
    def load_parameters(self, network_parameters):
        """
        # If you want to recompute few layers of the net in a sequential manner
        # change and load the parameters with this method and then rerun the net learning
        # (For example you computed a 2 layer network, and you want to rerun the second layer,
        # you use a new set of parameters, and rerun learning with rerun_layer=1, as the 
        # layers index start with 0.
        Net.load_parameters(network_parameters)
        Net.learn(dataset_train, dataset_test, rerun_layer = 1)
        """
        [[features_number, l1_norm_coeff, learning_rate, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, exploring], [learning_rate,\
        epochs, l1_norm_coeff, intermediate_dim_T, intermediate_dim_2D,\
        activity_th, batch_size, spacing_local_T]] = network_parameters
        
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        
        self.batch_size=batch_size
        self.intermediate_dim_T=intermediate_dim_T
        self.intermediate_dim_2D=intermediate_dim_2D
        self.activity_th = activity_th
        self.spacing_local_T = spacing_local_T
        self.learning_rate = learning_rate
        self.l1_norm_coeff = l1_norm_coeff
        self.epochs = epochs
        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.exploring = exploring

### Classifiers

    # Method for training a mlp classification model 
    # =============================================================================     
    from ._Classifiers_methods import mlp_classification_train

    # Method for testing the mlp classification model
    # =============================================================================     
    from ._Classifiers_methods import mlp_classification_test
    
    # Method for training a mlp-histogram classification model 
    # =============================================================================      
    from ._Classifiers_methods import hist_mlp_classification_train
    
    # Method for testing a mlp-histogram classification model 
    # =============================================================================      
    from ._Classifiers_methods import hist_mlp_classification_test
    
    # Method for training a lstm classification model 
    # =============================================================================      
    from ._Classifiers_methods import lstm_classification_train
   
    # Method for testing a lstm classification model 
    # =============================================================================      
    from ._Classifiers_methods import lstm_classification_test
    
    # Method for training a cnn classification model 
    # =============================================================================      
    from ._Classifiers_methods import cnn_classification_train
   
    # Method for testing a cnn classification model 
    # =============================================================================      
    from ._Classifiers_methods import cnn_classification_test
    
    # Method for testing an histogram classificator  
    # =============================================================================      
    from ._Classifiers_methods import hist_classification_test

### Plot methods
    
    # Method to plot reconstructed and original surfaces
    # =============================================================================
    from ._Plotting_methods import plt_surfaces_vs_reconstructions
    
    # Method to print loss history of the network
    # =============================================================================
    from ._Plotting_methods import plt_loss_history
    
    # Method to plot last layer activation of the network
    # =============================================================================
    from ._Plotting_methods import plt_last_layer_activation
    
    # Method to plot reverse activation of a sublayer output (output related to input)
    # =============================================================================
    from ._Plotting_methods import plt_reverse_activation