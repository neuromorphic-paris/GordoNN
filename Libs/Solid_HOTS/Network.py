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
from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.decomposition import IncrementalPCA
# from sklearn.preprocessing import StandardScaler
# import hdbscan

# Homemade Fresh Libraries like Grandma does
from Libs.Solid_HOTS._General_Func import local_surface_plot, surfaces_plot,\
     local_tv_generator, cross_tv_generator_conv, \
     cross_tv_generator, compute_abs_ind
     



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
    local_tv_length (list of int): the length of the Local time vector generated per each layer
    input_channels (int) : the total number of channels of the cochlea in the input files 
    taus_T(list of float lists) :  a list containing the time coefficient used for 
                                   the local time vector creations for each layer (first index)
                                   and each channel (second index). To keep it simple, 
                                   it's the result of a multiplication between a vector for each 
                                   layer(channel_taus) and a coefficient (taus_T_coeff).
    taus_2D (list of float) : a list containing the time coefficients used for the 
                             creation of timesurfaces per each layer
    activity_th (float) : The code will check that the sum(local surface)
    threads (int) : The network can compute timesurfaces in a parallel way,
                    this parameter set the number of multiple threads allowed to run
 

    verbose (boolean) : If True, the network will output messages to inform the 
                          the users about the current states as well as plots
    """
    def __init__(self, network_parameters):
        
        [[features_number, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, verbose], [
        n_batch_files, dataset_runs]] = network_parameters
            
                                                               
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        self.input_channels = input_channels       
        self.n_batch_files = n_batch_files
        self.dataset_runs  = dataset_runs

        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.verbose = verbose
      
    
    # =============================================================================  
    def learn_old(self, dataset, dataset_test, labels, labels_test, num_classes, rerun_layer=0):
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
            self.local_sublayer=[] # Kmeans for 0D Sublayers
            self.cross_sublayer=[] # Kmeans models for 2D Sublayers
            layers_index = np.arange(self.layers)
            self.net_local_response=[]
            self.net_0D_response_test=[]
            self.net_cross_response = []
            self.net_cross_response_test = []
            self.layer_dataset = []
            self.layer_dataset_test = []

            
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
                    layer_dataset=self.net_cross_response[layer-1]
                    layer_dataset_test=self.net_cross_response_test[layer-1]
                    self.layer_dataset=self.layer_dataset[:layer+1]
                    self.layer_dataset_test=self.layer_dataset_test[:layer+1]
              
                self.local_sublayer=self.local_sublayer[:layer]
                self.cross_sublayer=self.cross_sublayer[:layer]
                self.net_local_response = self.net_local_response[:layer]
                self.net_0D_response_test = self.net_0D_response_test[:layer]
                self.net_cross_response = self.net_cross_response[:layer]
                self.net_cross_response_test = self.net_cross_response_test[:layer]
                self.removed_ind = self.removed_ind[:layer]
                self.removed_ind_test = self.removed_ind_test[:layer]
                
            
                
            # GENERATING AND COMPUTING TIME LOCAL SURFACES RESPONSES
            if self.verbose is True:
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
            # if layer == 0 :
            # Generation of local surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads)(delayed(local_tv_generator)\
                                (recording, layer_dataset, self.polarities[layer],\
                                self.taus_T[layer], self.local_surface_length[layer],
                                self.verbose, self.activity_th, self.spacing_local_T[layer])\
                                for recording in range(len(layer_dataset)))
            # else:
            #     # Generation of local surfaces, computed on multiple threads
            #     results = Parallel(n_jobs=self.threads)(delayed(local_tv_generator_rate)\
            #                         (recording, layer_dataset, self.polarities[layer],\
            #                         self.taus_T[layer], self.local_surface_length[layer],
            #                         self.verbose, self.activity_th, self.spacing_local_T[layer])\
            #                         for recording in range(len(layer_dataset)))        
             
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
            # if layer == 0 :
            # Generation of local surfaces, computed on multiple threads                   
            results = Parallel(n_jobs=self.threads)(delayed(local_tv_generator)\
                                (recording, layer_dataset_test, self.polarities[layer],\
                                self.taus_T[layer], self.local_surface_length[layer],
                                self.verbose, self.activity_th, self.spacing_local_T[layer])\
                                for recording in range(len(layer_dataset_test)))
            # else:
            #     # Generation of local surfaces, computed on multiple threads
            #     results = Parallel(n_jobs=self.threads)(delayed(local_tv_generator_rate)\
            #                         (recording, layer_dataset_test, self.polarities[layer],\
            #                         self.taus_T[layer], self.local_surface_length[layer],
            #                         self.verbose, self.activity_th, self.spacing_local_T[layer])\
            #                         for recording in range(len(layer_dataset_test)))   
           
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
            if self.verbose is True:    
                print("Generating local_surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' LOCAL SURFACES FEATURES EXTRACTION ---')
                start_time = time.time()
            # batch_size=len(all_local_surfaces)//self.features_number[layer][1]  
            batch_size = 256*24 #Times the number of threads
            self.local_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][0], batch_size=batch_size).fit(all_local_surfaces))
            # kmeans = MiniBatchKMeans(n_clusters=self.features_number[layer][0], batch_size=batch_size)
            
            # n_batches=4
            # for i_batch in range(n_batches):
            #     kmeans.partial_fit(all_local_surfaces[i_batch::n_batches])
                
            # self.local_sublayer.append(kmeans)


            if self.verbose is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
                local_surface_plot(layer_dataset, local_surface_indices, all_local_surfaces, layer)
                local_surface_plot(layer_dataset_test, local_surface_indices_test, all_local_surfaces_test, layer)
                
                
            # Compute 0D sublayer activations
            # Train set
            current_pos = 0
            net_local_response = []
            for recording in range(len(layer_dataset)):                
                # Get network activations at steady state (after learning)
                recording_results = self.local_sublayer[layer].predict(np.array(all_local_surfaces[current_pos:current_pos+len(local_surface_indices[recording])]))                                
                current_pos += len(local_surface_indices[recording])
                net_local_response.append(events_from_activations(recording_results, local_surface_indices[recording], layer_dataset[recording]))
            # Test set            
            current_pos = 0
            net_0D_response_test = []
            for recording in range(len(layer_dataset_test)):                
                # Get network activations at steady state (after learning)
                recording_results = self.local_sublayer[layer].predict(np.array(all_local_surfaces_test[current_pos:current_pos+len(local_surface_indices_test[recording])]))
                current_pos += len(local_surface_indices_test[recording])
                net_0D_response_test.append(events_from_activations(recording_results, local_surface_indices_test[recording], layer_dataset_test[recording]))
            
            
            # clearing some variables
            gc.collect()
            
            # Save the 0D response
            self.net_local_response.append(net_local_response)
            self.net_0D_response_test.append(net_0D_response_test)
            
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.verbose is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            
            # Temporary lists used to store all the local surfaces produced here one for each class
            all_surfaces=[] 
            surface_indices=[]
            all_surfaces_test=[]
            surface_indices_test=[]
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads)(delayed(cross_tv_generator)\
                                                                  (recording,
                                                                   net_local_response, 
                                                                   self.polarities[layer],
                                                                   self.features_number[layer],
                                                                   self.taus_2D[layer],
                                                                   self.verbose)\
                                    for recording in range(len(net_local_response)))
           
            # unpacking the results                   
            for recording in range(len(net_local_response)):                    
                all_surfaces.append(results[recording][1])
                surface_indices.append(results[recording][0])
           
            all_surfaces = np.concatenate(all_surfaces, axis=0) 

            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads)(delayed(cross_tv_generator)\
                                                                  (recording,
                                                                   net_0D_response_test, 
                                                                   self.polarities[layer],
                                                                   self.features_number[layer],
                                                                   self.taus_2D[layer],
                                                                   self.verbose)\
                                    for recording in range(len(net_0D_response_test)))
           
            # unpacking the results                   
            for recording in range(len(net_0D_response_test)):                    
                all_surfaces_test.append(results[recording][1])
                surface_indices_test.append(results[recording][0])
                
            all_surfaces_test = np.concatenate(all_surfaces_test, axis=0) 
            
            if self.verbose is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            
            # The final results of the local surfaces test dataset computation
            all_surfaces = np.array(all_surfaces, dtype='float16') 
            all_surfaces_test = np.array(all_surfaces_test, dtype='float16')
            gc.collect()
            
            
            # # TRAINING CROSS SURFACES FEATURES
            # print('PCA')

            # all_surfaces_scaled = all_surfaces
            # all_surfaces_test_scaled = all_surfaces_test
            # pca_surf=IncrementalPCA(batch_size=50000)
            # surf_embedd=pca_surf.fit(all_surfaces_scaled)
            # exp_var=pca_surf.explained_variance_ratio_
            # n_dims = [i  for i in range(len(exp_var)) if sum(exp_var[:i])>0.95][0]
            # surf_embedd = pca_surf.transform(all_surfaces_scaled)[:,:n_dims]
            # self.PCA=pca_surf
            # print('PCA_DONE selected: '+str(n_dims)+' dimensions')
            # self.dimensions_selected = n_dims
            # batch_size=len(surf_embedd)//self.features_number[layer][1]
            # self.cross_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][1], batch_size=batch_size).fit(surf_embedd))
            # batch_size=len(all_surfaces)//self.features_number[layer][1]
            batch_size = 256*24*1000#Times the number of threads
            # self.cross_sublayer.append(MiniBatchKMea0ns(verbose=2, n_clusters=self.features_number[layer][1], batch_size=batch_size).fit(all_surfaces))
            kmeans2d = MiniBatchKMeans(n_clusters=self.features_number[layer][1], batch_size=batch_size, verbose=True)
            
            # n_batches=64#On_OFF
            n_batches=2048#FUll_data

            n_k_runs = 5
            old_inertia = 0
            partial_initail = 8
            all_surfaces_shuffl = np.random.permutation(all_surfaces) 
            try:
                for run_i in range(n_k_runs):
                    for i_batch in range(n_batches):
                        if run_i==-1 and i_batch==-1:
                            kmeans2d.fit(np.random.permutation(all_surfaces)[0::partial_initail])
                        else:
                            kmeans2d.partial_fit(np.random.permutation(all_surfaces)[i_batch::n_batches])
                        batch_distances = kmeans2d.transform(all_surfaces_shuffl[i_batch::n_batches])**2
                        batch_labels = kmeans2d.predict(all_surfaces_shuffl[i_batch::n_batches])
                        inertia = np.mean(batch_distances[(np.arange(len(batch_labels)), batch_labels)])
                        print("Partial fit of %4i out of %i" % (i_batch+run_i*n_batches, n_batches * n_k_runs))
                        print("Batch Inertia = %6f" % (inertia))
                        if np.abs(inertia-old_inertia) <= 0.0001:
                            raise StopIteration
            except StopIteration:
                print("Stopped for lack of improvement")
                pass
                    
            self.cross_sublayer.append(kmeans2d)
            
            
            if self.verbose is True:
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
                # surf_embedd=pca_surf.transform(np.array(all_surfaces_scaled[current_pos:current_pos+len(surface_indices[recording])]))[:,:n_dims]
                recording_results=self.cross_sublayer[layer].predict(np.array(all_surfaces[current_pos:current_pos+len(surface_indices[recording])]))
                # recording_results = self.cross_sublayer[layer].predict(surf_embedd)
                current_pos += len(surface_indices[recording])
                layer_2D_activations.append([net_local_response[recording][0][surface_indices[recording]], recording_results])
            del net_local_response

            # Test set            
            for recording in range(len(surface_indices_test)):
                # surf_embedd=pca_surf.transform(np.array(all_surfaces_test_scaled[current_pos_test:current_pos_test+len(surface_indices_test[recording])]))[:,:n_dims]
                recording_results=self.cross_sublayer[layer].predict(np.array(all_surfaces_test[current_pos_test:current_pos_test+len(surface_indices_test[recording])]))
                # recording_results = self.cross_sublayer[layer].predict(surf_embedd)
                current_pos_test += len(surface_indices_test[recording])
                layer_2D_activations_test.append([net_0D_response_test[recording][0][surface_indices_test[recording]], recording_results])
            del net_0D_response_test
            self.surfaces=[all_local_surfaces, all_local_surfaces_test, all_surfaces, all_surfaces_test]
            self.net_cross_response.append(layer_2D_activations)
            self.net_cross_response_test.append(layer_2D_activations_test)
            
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

    # =============================================================================  
    def learn(self, dataset, rerun_layer=0):
        """
        Network learning method. It saves net responses recallable with 
        method .last_layer_activity, so it can be used to train the classifiers.
        
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner

            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            self.local_sublayer=[] # Kmeans for 0D Sublayers
            self.cross_sublayer=[] # Kmeans models for 2D Sublayers
            layers_index = np.arange(self.layers)
            self.net_local_response=[]
            self.net_cross_response = []
            self.layer_dataset = []

        else:
            layers_index = np.arange(rerun_layer,self.layers)

                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset.append(layer_dataset)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset=self.net_cross_response[layer-1]
                    self.layer_dataset=self.layer_dataset[:layer+1]
              
                self.local_sublayer=self.local_sublayer[:layer]
                self.cross_sublayer=self.cross_sublayer[:layer]
                self.net_local_response = self.net_local_response[:layer]
                self.net_cross_response = self.net_cross_response[:layer]
                
            
            
            self.local_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][0], verbose=self.verbose))  
            self.local_batch_learning(layer_dataset, layer)
            self.net_local_response.append(self.local_batch_infering(layer_dataset, layer))
            # self.net_local_response.append(self.local_batch_infering_mod(layer_dataset, layer))
            
            self.cross_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][1], verbose=self.verbose))
            self.cross_batch_learning(self.net_local_response, layer)
            self.net_cross_response.append(self.cross_batch_infering(self.net_local_response, layer))
            
          
            
            layer_dataset = self.net_cross_response[-1]


                        
            # clearing some rubbish
            gc.collect()
            
  
        self.last_layer_activity = self.net_cross_response[-1]
        
        
    # =============================================================================  
    def learn_mod(self, dataset, rerun_layer=0):
        """
        Network learning method. It saves net responses recallable with 
        method .last_layer_activity, so it can be used to train the classifiers.
        
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner

            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            self.local_sublayer=[] # Kmeans for 0D Sublayers
            self.cross_sublayer=[] # Kmeans models for 2D Sublayers
            layers_index = np.arange(self.layers)
            self.net_local_response=[]
            self.net_cross_response = []
            self.layer_dataset = []

        else:
            layers_index = np.arange(rerun_layer,self.layers)

                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset.append(layer_dataset)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset=self.net_cross_response[layer-1]
                    self.layer_dataset=self.layer_dataset[:layer+1]
              
                self.local_sublayer=self.local_sublayer[:layer]
                self.cross_sublayer=self.cross_sublayer[:layer]
                self.net_local_response = self.net_local_response[:layer]
                self.net_cross_response = self.net_cross_response[:layer]
                
            
            
            self.net_local_response.append(self.local_batch_infering_mod(layer_dataset, layer))
            
            self.cross_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][1], verbose=self.verbose))
            self.cross_batch_learning(self.net_local_response, layer)
            self.net_cross_response.append(self.cross_batch_infering(self.net_local_response, layer))
            
          
            
            layer_dataset = self.net_cross_response[-1]


                        
            # clearing some rubbish
            gc.collect()
            
  
        self.last_layer_activity = self.net_cross_response[-1]

    # =============================================================================  
    def learn_conv(self, dataset, rerun_layer=0):
        """
        Network learning method. It saves net responses recallable with 
        method .last_layer_activity, so it can be used to train the classifiers.
        
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner

            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            self.local_sublayer=[] # Kmeans for 0D Sublayers
            self.cross_sublayer=[] # Kmeans models for 2D Sublayers
            layers_index = np.arange(self.layers)
            self.net_local_response=[]
            self.net_cross_response = []
            self.layer_dataset = []

        else:
            layers_index = np.arange(rerun_layer,self.layers)

                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset.append(layer_dataset)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset=self.net_cross_response[layer-1]
                    self.layer_dataset=self.layer_dataset[:layer+1]
              
                self.local_sublayer=self.local_sublayer[:layer]
                self.cross_sublayer=self.cross_sublayer[:layer]
                self.net_local_response = self.net_local_response[:layer]
                self.net_cross_response = self.net_cross_response[:layer]
                
            
            
            self.local_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][0], verbose=self.verbose))  
            self.local_batch_learning(layer_dataset, layer)
            self.net_local_response.append(self.local_batch_infering(layer_dataset, layer))
            # self.net_local_response.append(self.local_batch_infering_mod(layer_dataset, layer))
            
            self.cross_sublayer.append(MiniBatchKMeans(n_clusters=self.features_number[layer][1], verbose=self.verbose))
            self.cross_batch_learning_conv(self.net_local_response, layer)
            self.net_cross_response.append(self.cross_batch_infering_conv(self.net_local_response, layer))
            
          
            
            layer_dataset = self.net_cross_response[-1]


                        
            # clearing some rubbish
            gc.collect()
            
  
        self.last_layer_activity = self.net_cross_response[-1]
        

    # =============================================================================  
    def infer(self, dataset, rerun_layer=0):
        """
        Network infer method. It saves net responses recallable with 
        method .last_layer_activity_test, so it can be used to test the classifiers.
        
        Arguments:
            dataset (nested lists) : the test dataset the features will match.
    
            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset_test = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            layers_index = np.arange(self.layers)
            self.net_local_response_test = []
            self.net_cross_response_test = []
            self.layer_dataset_test = []
    
        else:
            layers_index = np.arange(rerun_layer,self.layers)
    
                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset_test.append(layer_dataset_test)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset_test=self.net_cross_response_test[layer-1]
                    self.layer_dataset_test=self.layer_dataset_test[:layer+1]
              
    
                self.net_local_response_test = self.net_local_response_test[:layer]
                self.net_cross_response_test = self.net_cross_response_test[:layer]
                
            
            self.net_local_response_test.append(self.local_batch_infering(layer_dataset_test, layer))
            # self.net_local_response_test.append(self.local_batch_infering_mod(layer_dataset_test, layer))
            
            self.net_cross_response_test.append(self.cross_batch_infering(self.net_local_response_test, layer))
            
          
            
            layer_dataset_test = self.net_cross_response_test[-1]
    
    
                        
            # clearing some rubbish
            gc.collect()
        

        self.last_layer_activity_test = self.net_cross_response_test[-1]
        
    # =============================================================================  
    def infer_mod(self, dataset, rerun_layer=0):
        """
        Network infer method. It saves net responses recallable with 
        method .last_layer_activity_test, so it can be used to test the classifiers.
        
        Arguments:
            dataset (nested lists) : the test dataset the features will match.
    
            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset_test = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            layers_index = np.arange(self.layers)
            self.net_local_response_test = []
            self.net_cross_response_test = []
            self.layer_dataset_test = []
    
        else:
            layers_index = np.arange(rerun_layer,self.layers)
    
                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset_test.append(layer_dataset_test)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset_test=self.net_cross_response_test[layer-1]
                    self.layer_dataset_test=self.layer_dataset_test[:layer+1]
              
    
                self.net_local_response_test = self.net_local_response_test[:layer]
                self.net_cross_response_test = self.net_cross_response_test[:layer]
                
            
            self.net_local_response_test.append(self.local_batch_infering_mod(layer_dataset_test, layer))
            
            self.net_cross_response_test.append(self.cross_batch_infering(self.net_local_response_test, layer))
            
          
            
            layer_dataset_test = self.net_cross_response_test[-1]
    
    
                        
            # clearing some rubbish
            gc.collect()
        

        self.last_layer_activity_test = self.net_cross_response_test[-1]
       
    # =============================================================================  
    def infer_conv(self, dataset, rerun_layer=0):
        """
        Network infer method. It saves net responses recallable with 
        method .last_layer_activity_test, so it can be used to test the classifiers.
        
        Arguments:
            dataset (nested lists) : the test dataset the features will match.
    
            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset_test = dataset 
        
        #check if you only want to rerun a layer
        if rerun_layer == 0:
            layers_index = np.arange(self.layers)
            self.net_local_response_test = []
            self.net_cross_response_test = []
            self.layer_dataset_test = []
    
        else:
            layers_index = np.arange(rerun_layer,self.layers)
    
                    
        for layer in layers_index: 
            
            
            # Check if the attributes need to be overwritten or created anew
            # This part of the code is only managing attributes to rerun a layer 
            # Or to append the datasets to data managing attributes
            if rerun_layer == 0:
                self.layer_dataset_test.append(layer_dataset_test)
            else:
                if layer==rerun_layer:
                    # The 2D net response is organized to directly feed the 
                    # second layer
                    layer_dataset_test=self.net_cross_response_test[layer-1]
                    self.layer_dataset_test=self.layer_dataset_test[:layer+1]
              
    
                self.net_local_response_test = self.net_local_response_test[:layer]
                self.net_cross_response_test = self.net_cross_response_test[:layer]
                
            
            self.net_local_response_test.append(self.local_batch_infering(layer_dataset_test, layer))
            # self.net_local_response_test.append(self.local_batch_infering_mod(layer_dataset_test, layer))
            
            self.net_cross_response_test.append(self.cross_batch_infering_conv(self.net_local_response_test, layer))
            
          
            
            layer_dataset_test = self.net_cross_response_test[-1]
    
    
                        
            # clearing some rubbish
            gc.collect()
        

        self.last_layer_activity_test = self.net_cross_response_test[-1]

    def local_batch_learning(self, layer_dataset, layer) :
        """
        Internal method to process and learn local features batch_wise
        """
        
        n_files = len(layer_dataset)
        n_batches=int(np.ceil(n_files/self.n_batch_files)) # number of batches per run
        n_runs = self.dataset_runs # how many time a single dataset get cycled.
        total_batches  = n_batches*n_runs
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' LOCAL TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                data_subset = layer_dataset[i_batch_run*self.n_batch_files:(i_batch_run+1)*self.n_batch_files]
                
                # Generation of local surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(local_tv_generator)\
                                    (data_subset[recording], self.polarities[layer],\
                                    self.taus_T[layer], self.local_surface_length[layer])\
                                    for recording in range(len(data_subset)))     
            
                # The final results of the local surfaces train dataset computation
                batch_local_tv = np.concatenate(results, axis=0)
                self.local_sublayer[layer].partial_fit(batch_local_tv)
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,total_batches,expected_t))                
                    batch_start_time = time.time()
            

        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))
            
        
    def cross_batch_learning(self, net_local_response, layer): 
        """
        Internal method to process and learn cross features batch_wise
        """
        
        n_files = len(net_local_response[layer])
        n_batches=int(np.ceil(n_files/self.n_batch_files)) # number of batches per run
        n_runs = self.dataset_runs # how many time a single dataset get cycled.
        total_batches  = n_batches*n_runs
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        for run in range(n_runs):        
            for i_batch_run in range(n_batches):
                
                data_subset = net_local_response[layer][i_batch_run*self.n_batch_files:(i_batch_run+1)*self.n_batch_files]
                
                # Generation of cross surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(cross_tv_generator)\
                                    (data_subset[recording], self.polarities[layer],\
                                    self.features_number[layer], self.taus_2D[layer])\
                                    for recording in range(len(data_subset)))
                    

              
                # results = []
                # for recording in range(len(data_subset)):
                #     results.append(cross_tv_generator(data_subset[recording], self.polarities[layer],\
                #     self.features_number[layer], self.taus_2D[layer]))
    
    
                # The final results of the local surfaces train dataset computation
                batch_cross_tv = np.concatenate(results, axis=0)
                self.cross_sublayer[layer].partial_fit(batch_cross_tv)
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,total_batches,expected_t))         
                    # batch_distances = self.cross_sublayer[layer].transform(batch_cross_tv)**2
                    # batch_labels = self.cross_sublayer[layer].predict(batch_cross_tv)
                    # inertia = np.mean(batch_distances[(np.arange(len(batch_labels)), batch_labels)])
                    # print("Batch Inertia = %6f" % (inertia))
                    batch_start_time = time.time()
            

        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))        
    
        
    def conv_cross_batch_learning(self, net_local_response, layer): 
        """
        Internal method to process and learn cross features batch_wise
        """
        
        n_files = len(net_local_response[layer])
        n_batches=int(np.ceil(n_files/self.n_batch_files)) # number of batches per run
        n_runs = self.dataset_runs # how many time a single dataset get cycled.
        total_batches  = n_batches*n_runs
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        for run in range(n_runs):        
            for i_batch_run in range(n_batches):
                
                data_subset = net_local_response[layer][i_batch_run*self.n_batch_files:(i_batch_run+1)*self.n_batch_files]
                
                # Generation of cross surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(cross_tv_generator_conv)\
                                    (data_subset[recording], self.polarities[layer],\
                                    self.features_number[layer], self.cross_surface_width[layer],\
                                    self.taus_2D[layer])\
                                    for recording in range(len(data_subset)))
                    

              
                # results = []
                # for recording in range(len(data_subset)):
                #     results.append(cross_tv_generator(data_subset[recording], self.polarities[layer],\
                #     self.features_number[layer], self.taus_2D[layer]))
    
    
                # The final results of the local surfaces train dataset computation
                batch_cross_tv = np.concatenate(results, axis=0)
                self.cross_sublayer[layer].partial_fit(batch_cross_tv)
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,total_batches,expected_t))         
                    # batch_distances = self.cross_sublayer[layer].transform(batch_cross_tv)**2
                    # batch_labels = self.cross_sublayer[layer].predict(batch_cross_tv)
                    # inertia = np.mean(batch_distances[(np.arange(len(batch_labels)), batch_labels)])
                    # print("Batch Inertia = %6f" % (inertia))
                    batch_start_time = time.time()
            

        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))              
    
    def local_batch_infering(self, layer_dataset, layer) :
        """
        Internal method to generate the response of a local features layer batch_wise
        """
        
        n_files = len(layer_dataset)
        n_batches=int(np.ceil(n_files/self.n_batch_files))
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' LOCAL TIME VECTORS GENERATION ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        local_response=[]    
        for i_batch in range(n_batches):
            
            data_subset = layer_dataset[i_batch*self.n_batch_files:(i_batch+1)*self.n_batch_files]
            
            # Generation of local surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(local_tv_generator)\
                                (data_subset[recording], self.polarities[layer],\
                                self.taus_T[layer], self.local_surface_length[layer])\
                                for recording in range(len(data_subset)))   
            
                
            for i_result in range(len(results)):
                batch_response=self.local_sublayer[layer].predict(results[i_result])
                local_response.append([data_subset[i_result][0], data_subset[i_result][1], batch_response])

            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                expected_t = batch_time*(n_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,n_batches,expected_t))               
                batch_start_time = time.time()                
                
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))                
                        
                        
        return local_response
        
    def local_batch_infering_mod(self, layer_dataset, layer) :
        """
        Internal method to generate the response of a local features layer batch_wise
        """
        
        n_files = len(layer_dataset)
        n_batches=int(np.ceil(n_files/self.n_batch_files))
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' LOCAL TIME VECTORS GENERATION ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        local_response=[]    
        for i_batch in range(n_batches):
            
            data_subset = layer_dataset[i_batch*self.n_batch_files:(i_batch+1)*self.n_batch_files]
            
 
            for i_result in range(len(data_subset)):
                local_response.append([data_subset[i_result][0], data_subset[i_result][1], np.zeros(len(data_subset[i_result][0]),dtype=int)])

            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                expected_t = batch_time*(n_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,n_batches,expected_t))               
                batch_start_time = time.time()                
                
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))                
                        
                        
        return local_response
    
    def cross_batch_infering(self, net_local_response, layer):  
        """
        Internal method to generate the response of a cross features layer batch_wise
        """
        
        n_files = len(net_local_response[layer])
        n_batches=int(np.ceil(n_files/self.n_batch_files))
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS GENERATION ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        cross_response=[]    
        for i_batch in range(n_batches):
            
            data_subset = net_local_response[layer][i_batch*self.n_batch_files:(i_batch+1)*self.n_batch_files]
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(cross_tv_generator)\
                                (data_subset[recording], self.polarities[layer],\
                                self.features_number[layer], self.taus_2D[layer])\
                                for recording in range(len(data_subset)))  
            
                
            for i_result in range(len(results)):
                batch_response=self.cross_sublayer[layer].predict(results[i_result])
                cross_response.append([data_subset[i_result][0],batch_response])

            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                expected_t = batch_time*(n_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,n_batches,expected_t))               
                batch_start_time = time.time()                
                
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))                
                        
                        
        return cross_response
    
    def conv_cross_batch_infering(self, net_local_response, layer):  
        """
        Internal method to generate the response of a cross features layer batch_wise
        """
        
        n_files = len(net_local_response[layer])
        n_batches=int(np.ceil(n_files/self.n_batch_files))
        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS GENERATION ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        cross_response=[]    
        for i_batch in range(n_batches):
            
            data_subset = net_local_response[layer][i_batch*self.n_batch_files:(i_batch+1)*self.n_batch_files]
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(cross_tv_generator_conv)\
                                (data_subset[recording], self.polarities[layer],\
                                self.features_number[layer], self.cross_surface_width[layer],\
                                self.taus_2D[layer])\
                                for recording in range(len(data_subset)))  
            
                
            for i_result in range(len(results)):
                batch_response=self.cross_sublayer[layer].predict(results[i_result])
                cross_response.append([data_subset[i_result][0],batch_response])

            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                expected_t = batch_time*(n_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,n_batches,expected_t))               
                batch_start_time = time.time()                
                
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))                
                        
                        
        return cross_response
    
    def cross_batch_tsMI_infering(self, net_local_response, layer, clusters_ind):  
        """
        Internal method to generate the response of a cross features layer batch_wise
        """
        
        n_files = len(net_local_response[layer])
        n_batches=int(np.ceil(n_files/self.n_batch_files))        
        #Set the verbose parameter for the parallel function.
        if self.verbose:
            par_verbose = 0
            print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS GENERATION ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        cross_response=[] 
        timesurfaces = []
        for i_batch in range(n_batches):
            
            data_subset = net_local_response[layer][i_batch*self.n_batch_files:(i_batch+1)*self.n_batch_files]
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.threads, verbose=par_verbose)(delayed(cross_tv_generator)\
                                (data_subset[recording], self.polarities[layer],\
                                self.features_number[layer], self.taus_2D[layer])\
                                for recording in range(len(data_subset)))  
            
                
            for i_result in range(len(results)):
                batch_response=self.cross_sublayer[layer].predict(results[i_result])
                ts_indx = np.ones(len(batch_response))
                for cluster_i in clusters_ind:
                    ts_indx *= batch_response!=cluster_i #it set at 0 
                timesurfaces.append(results[i_result][ts_indx==0])
                cross_response.append([data_subset[i_result][0][ts_indx==0],batch_response[ts_indx==0]])
                
            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                expected_t = batch_time*(n_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left " %(i_batch+1,n_batches,expected_t))               
                batch_start_time = time.time()                
                
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))                
                        
                        
        return cross_response, timesurfaces

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
        [[features_number, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, verbose], [
        activity_th, spacing_local_T]] = network_parameters
        
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        

        self.activity_th = activity_th
        self.spacing_local_T = spacing_local_T
        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.verbose = verbose
        
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
        [[features_number, local_surface_length,\
        input_channels, taus_T, taus_2D, threads, verbose], [
        activity_th, spacing_local_T]] = network_parameters
        
        # Setting network parameters as attributes
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.local_surface_length = local_surface_length
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        

        self.activity_th = activity_th
        self.spacing_local_T = spacing_local_T

        
        for layer in range(self.layers-1): # It's the number of different signals 
                                           # the 0D sublayer is receiveing
            self.polarities.append(features_number[layer][1])

        self.threads=threads
        self.verbose = verbose

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