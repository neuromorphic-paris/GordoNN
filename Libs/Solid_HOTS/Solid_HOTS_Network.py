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
import numpy as np 
import keras
from scipy.spatial import distance 
from sklearn.cluster import KMeans
from joblib import Parallel, delayed 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import gc

# Homemade Fresh Libraries like Gandma taught
from Libs.Solid_HOTS.Context_Surface_generators import Time_context, Time_context_later, Time_Surface
from Libs.Solid_HOTS.Solid_HOTS_Libs import create_vae, create_mlp, events_from_activations_T, events_from_activations_2D, context_plot, surfaces_plot



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
                                    is 0 for the centers of the 0D sublayer, and 1 for 
                                    the 2D centers
        context_lengths (list of int): the length of the time context generatef per each layer
        input_channels (int) : the total number of channels of the cochlea in the input files 
        taus_T(list of float lists) :  a list containing the time coefficient used for 
                                       the context creations for each layer (first index)
                                       and each channel (second index) 
        taus_2D (list of float) : a list containing the time coefficients used for the 
                                 creation of timesurfaces per each layer
        threads (int) : The network can compute timesurfaces in a parallel way,
                        this parameter set the number of multiple threads allowed to run
        exploring (boolean) : If True, the network will output messages to inform the 
                              the users about the current states and will save the 
                              basis at each update to build evolution plots (currently not 
                              available cos the learning is offline)
    """
    def __init__(self, features_number, context_lengths, input_channels, taus_T, taus_2D, 
                 threads=1, exploring=False):

        self.vaes_T=[]
        self.vaes_2D=[]
        self.taus_T = taus_T
        self.taus_2D = taus_2D
        self.layers = len(features_number)
        self.context_lengths = context_lengths
        self.features_number = features_number
        self.polarities = []
        self.polarities.append(input_channels)
        for layer in range(self.layers-1): # the number of chanels are considered as 
                                           # as polarities of a first layer
                                           # therefore self.polaroties is long
                                           # as the number of layers+1
            self.polarities.append(features_number[layer][1])
        self.threads=threads
        self.exploring = exploring
        #TODO decide wether this thing is worth the pain 
        ## Set of attributes used only in exploring mode
        if exploring is True:
            # attribute containing all 2D surfaces computed during a run in each layer 
            self.surfaces = []
            # attribute containing all 1D contexts computed during a run in each layer
            self.contexts = []
            # attribute containing all optimization errors computed during a run in each layer 
            self.errors = []
       
        
    # =============================================================================  
    def learn(self, dataset, learning_rate, epochs, l1_norm_coeff):
        """
        Network learning method, for now it based on kmeans and it is offline
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner
            learning_rate (float) : It's the learning rate of ADAM online optimization                 
        """
        
        layer_dataset = dataset
        first_sublayer=True
        # The code is going to run on gpus, to improve performances rather than 
        # a pure online algorithm I am going to minibatch 
        self.batch_size = 500
        del dataset
        for layer in range(self.layers):

            
            # Create the varational autoencoder for this layer
            intermediate_dim = 20
            self.vaes_T.append(create_vae(self.context_lengths[layer],
                                          self.features_number[layer][0],
                                          intermediate_dim, learning_rate[layer][0], l1_norm_coeff[layer][0], first_sublayer))
            first_sublayer=False
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' CONTEXTS GENERATION ---')
                start_time = time.time()
            if layer == 0 :
                all_contexts=[]
                for recording in range(len(layer_dataset)):
                    n_batch = len(layer_dataset[recording][0])//self.batch_size
                    # Cut the excess data in the first layer : 
                    layer_dataset[recording][0]=layer_dataset[recording][0][:n_batch*self.batch_size]
                    layer_dataset[recording][1]=layer_dataset[recording][1][:n_batch*self.batch_size]
                   
    
                    all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context)(event_ind, layer_dataset[recording],
                                                              self.taus_T[layer][layer_dataset[recording][1][event_ind]],
                                                              self.context_lengths[layer]) for event_ind in range(n_batch*self.batch_size))
                    all_contexts+=all_contexts_recording
                    gc.collect()
                    if self.exploring is True:
                        print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
                all_contexts = np.array(all_contexts, dtype='float32')
                                    
            else:
                all_contexts=np.zeros([num_of_2D_events*self.features_number[layer-1][1],self.context_lengths[layer]],dtype='float16')
                count=0
                for recording in range(len(layer_dataset)):
                    n_batch = len(layer_dataset[recording][0])//self.batch_size                     
                    all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context_later)(event_ind, layer_dataset[recording],
                                                              self.taus_T[layer][0],
                                                              self.context_lengths[layer]) for event_ind in range(n_batch*self.batch_size))
                    all_contexts[count:(n_batch*self.batch_size*self.features_number[layer-1][1])+count]=np.concatenate(all_contexts_recording,0).tolist()
                    count+=n_batch*self.batch_size*self.features_number[layer-1][1]
                    gc.collect()
                    if self.exploring is True:
                        print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
            if self.exploring is True:    
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' CONTEXTS FEATURES EXTRACTION ---')
                start_time = time.time()
            # Training the features 
            self.vaes_T[layer][0].fit(all_contexts, shuffle=True,
                     epochs=epochs[layer][0], batch_size=self.batch_size,
                     validation_data=(all_contexts, None))
            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
                context_plot(layer_dataset, all_contexts, layer)
                
            # Obtain Net activations
            current_pos = 0
            if layer == 0 :
                net_T_response = []
                original_timestamps = []
                for recording in range(len(layer_dataset)):                
                    # Get network activations at steady state (after learning)
                    recording_results = self.vaes_T[layer][1].predict(np.array(all_contexts[current_pos:current_pos+len(layer_dataset[recording][0])]), batch_size=self.batch_size)
                    current_pos += len(layer_dataset[recording][0])
                    net_T_response.append(events_from_activations_T(recording_results, layer_dataset[recording]))
                    original_timestamps.append(net_T_response[-1][0])
            else:
                # Get network activations at steady state (after learning)
                recording_results = self.vaes_T[layer][1].predict(all_contexts, batch_size=self.batch_size)
#                all_surfaces=2*recording_results.reshape(num_of_2D_events,self.features_number[layer-1][1]*self.features_number[layer][0])
                all_surfaces=recording_results.reshape(num_of_2D_events,self.features_number[layer-1][1]*self.features_number[layer][0])

            ##################
            self.tmporig_c=all_contexts[0:20000]
            self.tmpcr_c=self.vaes_T[layer][2].predict(self.vaes_T[layer][1].predict(all_contexts[0:20000]))
            ##################
            
            # clearing some variables
            del all_contexts, layer_dataset
            gc.collect()

            
            # Create the varational autoencoder for this layer
            intermediate_dim = 20
            self.vaes_2D.append(create_vae(self.polarities[layer]*self.features_number[layer][0],
                                        self.features_number[layer][1], intermediate_dim, learning_rate[layer][1],  l1_norm_coeff[layer][1], first_sublayer))
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            # 2D timesurface dimension
            ydim,xdim = [self.polarities[layer], self.features_number[layer][0]]
            if layer==0:
                all_surfaces = []
                for recording in range(len(net_T_response)):
                    # As a single time surface is build on all polarities, there is no need to build a time 
                    # surface per each event with a different polarity and equal time stamp, thus only 
                    # a fraction of the events are extracted here
                    n_batch = len(net_T_response[recording][0])//self.batch_size               
                    reference_event_jump=self.features_number[layer][0]                  
                    all_surfaces_recording = Parallel(n_jobs=self.threads)(delayed(Time_Surface)(xdim, ydim, event_ind,
                                                      self.taus_2D[layer], net_T_response[recording],
                                                      minv=0.1) for event_ind in range(0, n_batch*self.batch_size, reference_event_jump))
                    ################################################
#                    all_surfaces += [2*surf for surf in all_surfaces_recording]
                    all_surfaces += all_surfaces_recording
                    ################################################
                    gc.collect()
                    if self.exploring is True:
                        print("\r","Surfaces generation :", (recording+1)/len(net_T_response)*100,"%", end="")
            
            if self.exploring is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            all_surfaces =np.array(all_surfaces, dtype='float32')
            # Training the features 
            self.vaes_2D[layer][0].fit(all_surfaces, shuffle=True,
                     epochs=epochs[layer][1], batch_size=self.batch_size,
                     validation_data=(all_surfaces, None))
            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
                surfaces_plot(all_surfaces, self.polarities[layer], self.features_number[layer][0])
            # Obtain Net activations
            layer_2D_activations=[]
            current_pos = 0
            if layer==0:
                for recording in range(len(net_T_response)):
                    recording_results = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(net_T_response[recording][0])//reference_event_jump]), batch_size=self.batch_size)
                    current_pos += len(net_T_response[recording][0])//reference_event_jump
                    #TODO timestamps in the first entry of the list and then the results without explicit polarity (fix this later)
                    layer_2D_activations.append([net_T_response[recording][0][range(0,len(net_T_response[recording][0]),reference_event_jump)],recording_results])
                del net_T_response
            else:
                for recording in range(len(original_timestamps)):
                    recording_results = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(original_timestamps[recording])//reference_event_jump]), batch_size=self.batch_size)
                    current_pos += len(original_timestamps[recording])//reference_event_jump
                    #TODO timestamps in the first entry of the list and then the results without explicit polarity (fix this later)
                    layer_2D_activations.append([original_timestamps[recording][range(0,len(original_timestamps[recording]),reference_event_jump)],recording_results])
            layer_dataset = layer_2D_activations
            num_of_2D_events=current_pos
            
            ##################
            self.tmporig=all_surfaces[0:20000]
            self.tmpcr=self.vaes_2D[layer][2].predict(self.vaes_2D[layer][1].predict(all_surfaces[0:20000]))
            ##################
            
            # clearing some variables
            del all_surfaces
        self.last_layer_activity = layer_2D_activations

    # =============================================================================  
    def learn_old(self, dataset, learning_rate):
        """
        Network learning method, for now it based on kmeans and it is offline
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner
            learning_rate (float) : It's the learning rate of ADAM online optimization                 
        """
        
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            # The code is going to run on gpus, to improve performances rather than 
            # a pure online algorithm I am going to minibatch 
            self.batch_size = 500
            
            # Create the varational autoencoder for this layer
            intermediate_dim = 20
            self.vaes_T.append(create_vae(self.context_lengths[layer],
                                          self.features_number[layer][0],
                                          intermediate_dim, learning_rate[layer][0]))
            
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' CONTEXTS GENERATION ---')
                start_time = time.time()
            all_contexts=[]
            for recording in range(len(layer_dataset)):
                n_batch = len(layer_dataset[recording][0])//self.batch_size
                # Cut the excess data in the first layer : 
                if layer == 0 :
                    layer_dataset[recording][0]=layer_dataset[recording][0][:n_batch*self.batch_size]
                    layer_dataset[recording][1]=layer_dataset[recording][1][:n_batch*self.batch_size]
               

                all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context)(event_ind, layer_dataset[recording],
                                                          self.taus_T[layer][layer_dataset[recording][1][event_ind]],
                                                          self.context_lengths[layer]) for event_ind in range(n_batch*self.batch_size))
                all_contexts+=all_contexts_recording
                gc.collect()
                if self.exploring is True:
                    print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
                        
            if self.exploring is True:    
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' CONTEXTS FEATURES EXTRACTION ---')
                start_time = time.time()
            all_contexts =np.array(all_contexts)
            # Training the features 
            self.vaes_T[layer][0].fit(all_contexts, shuffle=False,
                     epochs=40, batch_size=self.batch_size,
                     validation_data=(all_contexts, None))
            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            net_T_response = []
            current_pos = 0
            for recording in range(len(layer_dataset)):                
                # Get network activations at steady state (after learning)
                recording_results, _, _ = self.vaes_T[layer][1].predict(np.array(all_contexts[current_pos:current_pos+len(layer_dataset[recording][0])]), batch_size=self.batch_size)
                current_pos += len(layer_dataset[recording][0])
                net_T_response.append(events_from_activations_T(recording_results, layer_dataset[recording]))

            # clearing some variables
            del layer_dataset, all_contexts
            
            # Create the varational autoencoder for this layer
            intermediate_dim = 20
            self.vaes_2D.append(create_vae(self.polarities[layer]*self.features_number[layer][0],
                                        self.features_number[layer][1], intermediate_dim, learning_rate[layer][1]))
            
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            # 2D timesurface dimension
            ydim,xdim = [self.polarities[layer], self.features_number[layer][0]]
            all_surfaces = []
            for recording in range(len(net_T_response)):
                # As a single time surface is build on all polarities, there is no need to build a time 
                # surface per each event with a different polarity and equal time stamp, thus only 
                # a fraction of the events are extracted here
                n_batch = len(net_T_response[recording][0])//self.batch_size               
                if layer !=0 :
                    reference_event_jump=self.features_number[layer][0]*self.features_number[layer-1][1]
                else:
                    reference_event_jump=self.features_number[layer][0]
                    
                all_surfaces_recording = Parallel(n_jobs=self.threads)(delayed(Time_Surface)(xdim, ydim, event_ind,
                                                  self.taus_2D[layer], net_T_response[recording],
                                                  minv=0.1) for event_ind in range(0, n_batch*self.batch_size, reference_event_jump))
                all_surfaces += all_surfaces_recording
                gc.collect()
                if self.exploring is True:
                    print("\r","Surfaces generation :", (recording+1)/len(net_T_response)*100,"%", end="")
            
            if self.exploring is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            all_surfaces =np.array(all_surfaces)
            # Training the features 
            self.vaes_2D[layer][0].fit(all_surfaces, shuffle=False,
                     epochs=40, batch_size=self.batch_size,
                     validation_data=(all_surfaces, None))
            if self.exploring is True:
                print("\n Features extraction took %s seconds." % (time.time() - start_time))
            # Obtain Net activations
            net_2D_response = []
            layer_2D_activations=[]
            current_pos = 0
            for recording in range(len(net_T_response)):
                recording_results, _, _ = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(net_T_response[recording][0])//reference_event_jump]), batch_size=self.batch_size)
                current_pos += len(net_T_response[recording][0])//reference_event_jump
                # Generate new events only if I am not at the last layer
                if layer != (self.layers-1):
                    net_2D_response.append(events_from_activations_2D(recording_results, [net_T_response[recording][0][range(0,len(net_T_response[recording][0]),reference_event_jump)],
                                                                      net_T_response[recording][1][range(0,len(net_T_response[recording][0]),reference_event_jump)]]))
                layer_2D_activations.append(recording_results)
            layer_dataset = net_2D_response
            # clearing some variables
            del net_T_response, all_surfaces
        self.last_layer_activity = layer_2D_activations
        
    # =============================================================================         
    def compute_response(self, dataset):
        """
        Method used to compute the full network response to a dataset.
        The last layer output is stored in .last_layer_activity
        Arguments:
            dataset : the input dataset for the network
        """
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' CONTEXTS GENERATION ---')
                start_time = time.time()
            if layer == 0 :
                all_contexts=[]
                for recording in range(len(layer_dataset)):
                    n_batch = len(layer_dataset[recording][0])//self.batch_size
                    # Cut the excess data in the first layer : 
                    layer_dataset[recording][0]=layer_dataset[recording][0][:n_batch*self.batch_size]
                    layer_dataset[recording][1]=layer_dataset[recording][1][:n_batch*self.batch_size]
                    
    
                    all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context)(event_ind, layer_dataset[recording],
                                                              self.taus_T[layer][layer_dataset[recording][1][event_ind]],
                                                              self.context_lengths[layer]) for event_ind in range(n_batch*self.batch_size))
                    all_contexts+=all_contexts_recording
                    gc.collect()
                    if self.exploring is True:
                        print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
                all_contexts = np.array(all_contexts)
                                    
            else:
                all_contexts=np.zeros([num_of_2D_events*self.features_number[layer-1][1],self.context_lengths[layer]],dtype='float16')
                count=0
                for recording in range(len(layer_dataset)):
                    n_batch = len(layer_dataset[recording][0])//self.batch_size                     
                    all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context_later)(event_ind, layer_dataset[recording],
                                                              self.taus_T[layer][0],
                                                              self.context_lengths[layer]) for event_ind in range(n_batch*self.batch_size))
                    all_contexts[count:(n_batch*self.batch_size*self.features_number[layer-1][1])+count]=np.concatenate(all_contexts_recording,0).tolist()
                    count+=n_batch*self.batch_size*self.features_number[layer-1][1]
                    gc.collect()
                    if self.exploring is True:
                        print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
                        
            if self.exploring is True:    
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' CONTEXTS RESPONSE COMPUTATION ---')
                start_time = time.time()
            all_contexts =np.array(all_contexts)
            # Obtain Net activations
            current_pos = 0
            if layer == 0 :
                net_T_response = []
                original_timestamps = []
                for recording in range(len(layer_dataset)):                
                    # Get network activations at steady state (after learning)
                    recording_results = self.vaes_T[layer][1].predict(np.array(all_contexts[current_pos:current_pos+len(layer_dataset[recording][0])]), batch_size=self.batch_size)
                    current_pos += len(layer_dataset[recording][0])
                    net_T_response.append(events_from_activations_T(recording_results, layer_dataset[recording]))
                    original_timestamps.append(net_T_response[-1][0])
            else:
                # Get network activations at steady state (after learning)
                recording_results = self.vaes_T[layer][1].predict(all_contexts, batch_size=self.batch_size)
                all_surfaces=recording_results.reshape(num_of_2D_events,self.features_number[layer-1][1]*self.features_number[layer][0])
           
            if self.exploring is True:
                print("\n Response computation took %s seconds." % (time.time() - start_time))
            # clearing some variables
            del layer_dataset, all_contexts
                        
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            # 2D timesurface dimension
            ydim,xdim = [self.polarities[layer], self.features_number[layer][0]]
            if layer==0:
                all_surfaces = []
                for recording in range(len(net_T_response)):
                    # As a single time surface is build on all polarities, there is no need to build a time 
                    # surface per each event with a different polarity and equal time stamp, thus only 
                    # a fraction of the events are extracted here
                    n_batch = len(net_T_response[recording][0])//self.batch_size               
                    reference_event_jump=self.features_number[layer][0]                  
                    all_surfaces_recording = Parallel(n_jobs=self.threads)(delayed(Time_Surface)(xdim, ydim, event_ind,
                                                      self.taus_2D[layer], net_T_response[recording],
                                                      minv=0.1) for event_ind in range(0, n_batch*self.batch_size, reference_event_jump))
                    all_surfaces += all_surfaces_recording
                    gc.collect()
                if self.exploring is True:
                    print("\r","Surfaces generation :", (recording+1)/len(net_T_response)*100,"%", end="")
            
            if self.exploring is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            all_surfaces =np.array(all_surfaces)
            layer_2D_activations=[]
            # Obtain Net activations
            current_pos = 0
            if layer==0:
                for recording in range(len(net_T_response)):
                    recording_results = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(net_T_response[recording][0])//reference_event_jump]), batch_size=self.batch_size)
                    current_pos += len(net_T_response[recording][0])//reference_event_jump
                    #TODO timestamps in the first entry of the list and then the results without explicit polarity (fix this later)
                    layer_2D_activations.append([net_T_response[recording][0][range(0,len(net_T_response[recording][0]),reference_event_jump)],recording_results])
                del net_T_response
            else:
                for recording in range(len(original_timestamps)):
                    recording_results = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(original_timestamps[recording])//reference_event_jump]), batch_size=self.batch_size)
                    current_pos += len(original_timestamps[recording])//reference_event_jump
                    #TODO timestamps in the first entry of the list and then the results without explicit polarity (fix this later)
                    layer_2D_activations.append([original_timestamps[recording][range(0,len(original_timestamps[recording]),reference_event_jump)],recording_results])
            layer_dataset = layer_2D_activations
            num_of_2D_events=current_pos
            if self.exploring is True:
                print("\n Response computation took %s seconds." % (time.time() - start_time))
            # clearing some variables
            del all_surfaces
        
        self.last_layer_activity = layer_2D_activations

    # =============================================================================         
    def compute_response_old(self, dataset):
        """
        Method used to compute the full network response to a dataset.
        The last layer output is stored in .last_layer_activity
        Arguments:
            dataset : the input dataset for the network
        """
        layer_dataset = dataset
        del dataset
        for layer in range(self.layers):
            # The code is going to run on gpus, to improve performances rather than 
            # a pure online algorithm I am going to minibatch 
            batch_size = 500
            
            # GENERATING AND COMPUTING TIME CONTEXT RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' CONTEXTS GENERATION ---')
                start_time = time.time()
            all_contexts=[]
            for recording in range(len(layer_dataset)):
                n_batch = len(layer_dataset[recording][0])//batch_size
                # Cut the excess data in the first layer : 
                if layer == 0 :
                    layer_dataset[recording][0]=layer_dataset[recording][0][:n_batch*batch_size]
                    layer_dataset[recording][1]=layer_dataset[recording][1][:n_batch*batch_size]
               
                all_contexts_recording = Parallel(n_jobs=self.threads)(delayed(Time_context)(event_ind, layer_dataset[recording],
                                                          self.taus_T[layer][layer_dataset[recording][1][event_ind]],
                                                          self.context_lengths[layer]) for event_ind in range(n_batch*batch_size))
                all_contexts+=all_contexts_recording
                gc.collect()
                if self.exploring is True:
                    print("\r","Contexts generation :", (recording+1)/len(layer_dataset)*100,"%", end="")
                        
            if self.exploring is True:    
                print("Generating contexts took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' CONTEXTS RESPONSE COMPUTATION ---')
                start_time = time.time()
            all_contexts =np.array(all_contexts)
            # Obtain Net activations
            net_T_response = []
            current_pos = 0
            for recording in range(len(layer_dataset)):   
                
                recording_results, _, _ = self.vaes_T[layer][1].predict(np.array(all_contexts[current_pos:current_pos+len(layer_dataset[recording][0])]), batch_size=batch_size)
                current_pos += len(layer_dataset[recording][0])
                net_T_response.append(events_from_activations_T(recording_results, layer_dataset[recording]))
           
            if self.exploring is True:
                print("\n Response computation took %s seconds." % (time.time() - start_time))
            # clearing some variables
            del layer_dataset, all_contexts
                        
            # GENERATING AND COMPUTING SURFACES RESPONSES
            if self.exploring is True:
                print('\n--- LAYER '+str(layer)+' SURFACES GENERATION ---')
            start_time = time.time()
            # 2D timesurface dimension
            ydim,xdim = [self.polarities[layer], self.features_number[layer][0]]
            all_surfaces = []
            for recording in range(len(net_T_response)):
                # As a single time surface is build on all polarities, there is no need to build a time 
                # surface per each event with a different polarity and equal time stamp, thus only 
                # a fraction of the events are extracted here
                n_batch = len(net_T_response[recording][0])//batch_size
                if layer !=0 :
                    reference_event_jump=self.features_number[layer][0]*self.features_number[layer-1][1]
                else:
                    reference_event_jump=self.features_number[layer][0]
                    
                all_surfaces_recording = Parallel(n_jobs=self.threads)(delayed(Time_Surface)(xdim, ydim, event_ind,
                                                  self.taus_2D[layer], net_T_response[recording],
                                                  minv=0.1) for event_ind in range(0, n_batch*batch_size, reference_event_jump))
                all_surfaces += all_surfaces_recording
                gc.collect()
                if self.exploring is True:
                    print("\r","Surfaces generation :", (recording+1)/len(net_T_response)*100,"%", end="")
            
            if self.exploring is True:
                print("Generating surfaces took %s seconds." % (time.time() - start_time))
                print('\n--- LAYER '+str(layer)+' SURFACES FEATURES EXTRACTION ---')
            start_time = time.time()
            all_surfaces =np.array(all_surfaces)
            layer_2D_activations=[]
            # Obtain Net activations
            net_2D_response = []
            current_pos = 0
            for recording in range(len(net_T_response)):
                recording_results, _, _ = self.vaes_2D[layer][1].predict(np.array(all_surfaces[current_pos:current_pos+len(net_T_response[recording][0])//reference_event_jump]), batch_size=batch_size)
                current_pos += len(net_T_response[recording][0])//reference_event_jump
                # Generate new events only if I am not at the last layer
                if layer != (self.layers-1):
                    net_2D_response.append(events_from_activations_2D(recording_results, [net_T_response[recording][0][range(0,len(net_T_response[recording][0]),reference_event_jump)],
                                                                                          net_T_response[recording][1][range(0,len(net_T_response[recording][0]),reference_event_jump)]]))
                layer_2D_activations.append(recording_results)        
            
            layer_dataset = net_2D_response
            if self.exploring is True:
                print("\n Response computation took %s seconds." % (time.time() - start_time))
            # clearing some variables
            del net_T_response, all_surfaces
        
        self.last_layer_activity = layer_2D_activations

    # Method for training a mlp classification model 
    # =============================================================================      
    def mlp_single_word_classification_train(self, classes, wordpos, number_of_labels, learning_rate, dataset=[]):
        """
        Method to train a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            labels (numpy array int) : array of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
            learning_rate (float) : The method is Adam 
            dataset (nested lists) : the dataset used for learn classification, if not declared the method
                         will use the last response of the network. To avoid surprises 
                         check that the labels inserted here and the dataset used for 
                         either .learn .full_net_dataset_response a previous .mlp_classification_train
                         .mlp_classification_test is matching (Which is extremely faster
                         than computing the dataset again, but be aware of that)

        """
        
        if dataset:
            self.compute_response(dataset=dataset)            
        last_layer_activity = self.last_layer_activity 
        num_of_recordings=len(last_layer_activity)
        begin = wordpos[0]
        end = wordpos[1]
        labels=[]
        for record in range(num_of_recordings):
            record_labels=np.zeros([len(last_layer_activity[record][0]),2])
            beg_ind=np.where(last_layer_activity[record][0]>=begin[record])[0][0]
            end_ind=np.where(last_layer_activity[record][0]<=end[record])[0][-1]
            record_labels[beg_ind:end_ind,classes[record]]=np.ones(end_ind-beg_ind)
            labels.append(record_labels)
        labels=np.concatenate(labels)
        last_layer_activity_concatenated = np.concatenate([last_layer_activity[recording][1] for recording in range(num_of_recordings)])
        n_latent_var = self.features_number[-1][-1]
        self.mlp = create_mlp(input_size=n_latent_var,hidden_size=10, output_size=number_of_labels, 
                              learning_rate=learning_rate)
        self.mlp.summary()
        self.mlp.fit(np.array(last_layer_activity_concatenated), labels,
          epochs=40,
          batch_size=self.batch_size, shuffle=True)
            
        if self.exploring is True:
            print("Training ended, you can now access the trained network with the method .mlp")

##Todo treat equal values
    # Method for testing the mlp classification model
    # =============================================================================      
    def mlp_single_word_classification_test(self, classes, number_of_labels, threshold, dataset=[]):
        """
        Method to test a simple mlp, to a classification task, to test the feature 
        rapresentation automatically extracted by HOTS
        
        Arguments:
            labels (list of int) : List of integers (labels) of the dataset
            number_of_labels (int) : The total number of different labels that,
                                     I am excepting in the dataset, (I know that i could
                                     max(labels) but, first, it's wasted computation, 
                                     second, the user should move his/her ass and eat 
                                     less donuts)
            dataset (nested lists) : the dataset used for testing classification, if not declared the method
                                     will use the last response of the network. To avoid surprises 
                                     check that the labels inserted here and the dataset used for 
                                     either .learn .full_net_dataset_response a previous .mlp_classification_train
                                     .mlp_classification_test is matching (Which is extremely faster
                                     than computing the dataset again, but be aware of that)

        """
        if dataset:
            self.compute_response(dataset=dataset)     
        last_layer_activity = self.last_layer_activity    
        num_of_recordings = len(last_layer_activity)
        last_layer_activity_concatenated = np.concatenate([last_layer_activity[recording][1] for recording in range(num_of_recordings)])
        predicted_labels_ev=self.mlp.predict(np.array(last_layer_activity_concatenated),batch_size=self.batch_size)
        counter=0
        predicted_labels=[]
        for recording in range(len(self.last_layer_activity)):
            tmp = predicted_labels_ev[counter:counter+len(self.last_layer_activity[recording][0])]
            tmp=(np.asarray(tmp)-0.5)
            new_threshold=threshold-0.5
            tmp=((abs(tmp)>new_threshold)*1>=1)*tmp
#            ones_sum = sum([tmp[i][1]>=threshold for i in range(len(tmp))])
#            zeros_sum = sum([tmp[i][0]>=threshold for i in range(len(tmp))])
#            predicted_labels.append(1*(ones_sum>=zeros_sum))
            off_sum = sum(tmp[:,0])
            on_sum = sum(tmp[:,1])
            predicted_labels.append(1*(on_sum>=off_sum))
            counter += len(self.last_layer_activity[recording][0])
        prediction_rate=0
        for i,true_label in enumerate(classes):
            prediction_rate += (predicted_labels[i] == true_label)/len(classes)
        return prediction_rate, predicted_labels, predicted_labels_ev
    
    # =============================================================================          
    def plot_vae_decode_2D(self, layer, sublayer, variables_ind, variable_fix=0):
        """
        Plots reconstructed timesurfaces as function of 2-dim latent vector
        of a selected layer of the network
        Arguments:
            layer (int) : layer used to locate the latent variable that will be 
                         used to decode the timesurfaces
            sublayer (int) : Decide whether to plot the Time contexts sublayer 
                             with "0" or the 2D one "1"
            variables_ind (list of 2 int) : as this method plot a 2D rapresentation
                                            it expects two latent variables to work
                                            with, thus here you can select the index
                                            of the latent variables that will 
                                            be displayed
            variable_fix (float) : the value at which all other latent variables
                                   will be fixed, 0 on default
        """      
        
        if sublayer==0:
            size_x = self.context_lengths[layer]
            size_y=1
            decoder = self.vaes_T[layer][2]
        else:
            size_x = self.features_number[layer][0]
            size_y = self.polarities[layer]
            decoder = self.vaes_2D[layer][2]
        
        # display a 30x30 2D manifold of timesurfaces
        n = 30
        figure = np.zeros((size_y * n, size_x*n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
    
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.ones(self.features_number[layer][sublayer])*variable_fix
                z_sample[variables_ind[0]]=xi
                z_sample[variables_ind[1]]=yi
                x_decoded = decoder.predict(np.array([z_sample]))
                tsurface = x_decoded[0].reshape(size_y, size_x)
                figure[i * size_y: (i + 1) * size_y,
                       j * size_x: (j + 1) * size_x] = tsurface
    
        plt.figure(figsize=(10, 10))
        plt.title("2D Latent decoding grid")
        start_range_x = size_x // 2
        end_range_x = n * size_x + start_range_x + 1
        pixel_range_x = np.arange(start_range_x, end_range_x, size_x)
        start_range_y = size_y // 2
        end_range_y = n * size_y + start_range_y + 1
        pixel_range_y = np.arange(start_range_y, end_range_y, size_y)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range_x, sample_range_x)
        plt.yticks(pixel_range_y, sample_range_y)
        plt.xlabel("z["+str(variables_ind[0])+"]")
        plt.ylabel("z["+str(variables_ind[1])+"]")
        plt.imshow(figure)
        plt.show()
    
    # =============================================================================             
    def plot_vae_space_2D(self, layer, variables_ind, label_names, labels, dataset=[]):
        """
        Plots latent rapresentation of a dataset for a given layer, or the latest
        network activations if no dataset is given,
        the data is also colored given the labels 
        Arguments:
            layer (int) : layer used to locate the latent variable that will be 
                           plotted, ignored if no dataset is given 
            variables_ind (list of 2 int) : as this method plot a 2D rapresentation
                                            it expects two latent variables to work
                                            with, thus here you can select the index
                                            of the latent variables that will 
                                            be displayed
            labels_names (list of strings) : The name of each label, used to plot 
                                             the legend
            labels (list of int) : List of integers (labels) of the dataset
            dataset (nested lists) : the dataset that will generate the responses
                                     through the .full_net_dataset_response method
                                     
                                     To avoid surprises 
                                     check that the labels inserted here and the dataset used for 
                                     either .learn .full_net_dataset_response a previous .mlp_classification_train
                                     .mlp_classification_test is matching (Which is extremely faster
                                     than computing the dataset again, but be aware of that)
        """
        if dataset:
            selected_layer_activity = self.full_net_dataset_response(dataset=dataset,layer_stop=layer)     
        else:
            selected_layer_activity = self.last_layer_activity   
        
        # Plot variable Space
        # display a 2D plot of the time surfaces in the latent space
        
        # Create a label array to specify the labes for each timesurface
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title("2D Latent dataset rapresentation")
        custom_lines = [Line2D([0], [0], color="C"+str(label), lw=1) for label in range(len(label_names))]
        for recording in range(len(labels)):
            ax.scatter(selected_layer_activity[recording][:, variables_ind[0]], selected_layer_activity[recording][:, variables_ind[1]], c="C"+str(labels[recording]))
        ax.legend(custom_lines,label_names)
        plt.xlabel("z["+str(variables_ind[0])+"]")
        plt.ylabel("z["+str(variables_ind[1])+"]")
        plt.show()
    

             ## ELEPHANT GRAVEYARD, WHERE ALL THE UNUSED METHODS GO TO SLEEP, ##
              ##  UNTIL A LAZY DEVELOPER WILL DECIDE WHAT TO DO WITH THEM    ##
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================  
        # =============================================================================
    # =============================================================================  



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
            norm_histograms[label,:] = histograms[label,:]/(spikes_per_label[label]+(spikes_per_label[label]==0)) #putting all the zeros to 1 if any
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
            norm_histograms[batch,:] = histograms[batch,:]/(len(net_response[batch][0])+(len(net_response[batch][0])==0)) #putting all the zeros to 1 if any
        
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