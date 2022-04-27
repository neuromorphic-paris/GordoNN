#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:03:43 2022

@author: marcorax93
"""

import numpy as np
import time
from joblib import Parallel, delayed 
from Cross_Layer import Cross_Layer, \
                        cross_tv_generator,\
                        cross_tv_generator_conv

class Cross_class_layer(Cross_Layer):
    """
    This is a modified Cross layer, that rather than clastering cross vectors,
    applyes a shallow classifier like an mlp on the vector to act as an 
    event based classifier. This can only be selected as the last layer of the 
    network.
    
    Cross Layer constructor arguments: 
        
    n_hidden_units (int) : the number of hidden units of the mlp.
                       
    cross_tv_width (int): the width of the cross time vectors across channels,
                          polarity, if set to None the vector always spans across 
                          all the available channels, and it is not centered
                          on the reference event anymore. IF NUMBER
                          IT HAS TO BE ODD
    
    n_input_channels (int): the total number of channels or polarities of the 
                           previous layer.
                                                      
    taus (float):  a list containing the time coefficient 
                   used for the local time vector creations 
                   for each channel of the cochlea, if it is
                   a single float, all channels/polarities
                   will have the same tau.
    
    n_input_features (int): the number of features of the previous layer, None,
                            if the previous layer didn't have any features 
                            (cochlea output)
    
    n_batch_files (int): the number of files processed for a single batch of 
                         learning/infering. If None or > than the number of files
                         in the dataset, data will be processed in a single run.
                         Clustering is performed bua minibatch_kmeans, if number
                         of batches >1 the quality of clustering might be worse.
                            
    dataset_runs (int): the number of runs for a single dataset, if data is run 
                        in a single batch, this parameter will be authomatically
                        set to 1.
    
    n_threads (int) : The network can compute timevectors in a parallel way,
                    this parameter set the number of the parallel jobs that 
                    it can use.
 
    verbose (boolean) : If True, the layer will output messages to inform the 
                          the users about the current states as well as plots.
    """
    def __init__(self, n_hidden_units, cross_tv_width, n_input_channels, 
                 taus, n_input_features=None, n_batch_files=None,
                 dataset_runs=1, n_threads=8, verbose=False):
        
        self.n_hidden_units = n_hidden_units
        self.cross_tv_width = cross_tv_width
        self.n_input_channels = n_input_channels
        self.taus = taus
        self.n_input_features = n_input_features
        self.n_batch_files = n_batch_files
        self.dataset_runs = dataset_runs
        self.n_threads = n_threads
        self.verbose = verbose
        
    def learn(self, layer_dataset):
        """
        Method to process and learn cross features.
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        
        Arguments: 
             layer_dataset: list of individual event based recoriding as
                            generated by the cochlea
                            
        """
        
        # Build the vector of individual taus
        if type(self.taus) is np.ndarray or type(self.taus) is list :
            taus = np.array(self.taus)
        else:
            taus = self.taus*np.ones(self.n_input_channels)
            
        # Check the runtime mode (multiple batches or single batch)
        n_files = len(layer_dataset)
        
        if self.n_batch_files==None:
            n_batches = 1
            n_runs = 1
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
        
        if n_batches==1:
            n_runs = 1
        else:
            n_runs = self.dataset_runs # how many time a single dataset get cycled.
        
        total_batches  = n_batches*n_runs
        
        #Check if it is in convolutional mode or not
        if self.cross_tv_width == None or self.cross_tv_width >= self.n_input_channels :
            #Full layer
            cross_width = self.n_input_channels
            conv = False
        else:
            #Convolutional mode
            cross_width = self.cross_tv_width
            conv = True
        
        #Check if previous layer had features or not
        if self.n_input_features == None:
            n_input_features = 1
        else:
            n_input_features = self.n_input_features
            
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
        
        #TRAIN AN MLP HERE    
        kmeans = MiniBatchKMeans(n_clusters=self.n_features,
                                 verbose=self.verbose)
        
        kmeans._n_threads = self.n_threads
        
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                rec_ind_1 = i_batch_run*n_batch_files
                rec_ind_2 = (i_batch_run+1)*n_batch_files

                data_subset = layer_dataset[rec_ind_1:rec_ind_2]
                
                # check if it is a convolutional layer.
                if conv:
                    
                    # Generation of cross surfaces, computed on multiple threads
                    results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                        (delayed(cross_tv_generator_conv)\
                                        (data_subset[recording], 
                                          self.n_input_channels,\
                                          n_input_features, cross_width,\
                                          self.taus)\
                                      for recording in range(len(data_subset)))
                    # results=[]
                    # for recording in range(len(data_subset)):
                    #     result = cross_tv_generator_conv(data_subset[recording],\
                    #               self.n_input_channels,\
                    #               n_input_features, cross_width,\
                    #               self.taus)
                    #     results.append(result)
                
                else:
                                  
                    #Generation of cross surfaces, computed on multiple threads
                    results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                        (delayed(cross_tv_generator)\
                                        (data_subset[recording], 
                                          self.n_input_channels,\
                                          n_input_features,self.taus)\
                                      for recording in range(len(data_subset)))
          
            
                # The final results of the local surfaces train dataset computation
                batch_local_tv = np.concatenate(results, axis=0)
                
                if n_batches==1:
                    kmeans.fit(batch_local_tv)
                else:
                    kmeans.partial_fit(batch_local_tv)
                
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left "\
                          %(i_batch+1,total_batches,expected_t))                
                    batch_start_time = time.time()
            

        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))
            
        self.features = kmeans.cluster_centers_
            
    def predict(self, layer_dataset):
        """
        Method to generate cross features response.
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        
        Arguments: 
            layer_dataset: list of individual event based recoriding as
                           generated by the cochlea
                          
        Returns: 
            cross_response: list of individual event based recording with a
                            new array of the corresponfing feature index
                            per each event. Any eventual previous feature
                            array is removed.
        """
        
        # Build the vector of individual taus
        if type(self.taus) is np.ndarray or type(self.taus) is list :
            taus = np.array(self.taus)
        else:
            taus = self.taus*np.ones(self.n_input_channels)
            
        # Check the runtime mode (multiple batches or single batch)
        n_files = len(layer_dataset)
        
        if self.n_batch_files==None:
            n_batches = 1
            n_runs = 1
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
        
        if n_batches==1:
            n_runs = 1
        else:
            n_runs = self.dataset_runs # how many time a single dataset get cycled.
        
        total_batches  = n_batches*n_runs
        
        #Check if it is in convolutional mode or not
        if self.cross_tv_width == None or self.cross_tv_width >= self.n_input_channels :
            #Full layer
            cross_width = self.n_input_channels
            conv = False
        else:
            #Convolutional mode
            cross_width = self.cross_tv_width
            conv = True
        
        #Check if previous layer had features or not
        if self.n_input_features == None:
            n_input_features = 1
        else:
            n_input_features = self.n_input_features
            
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
          
        kmeans = MiniBatchKMeans(n_clusters=self.n_features,
                                 verbose=self.verbose)
        
        kmeans._n_threads = self.n_threads
        kmeans.cluster_centers_ = self.features
        
        cross_response=[]
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                rec_ind_1 = i_batch_run*n_batch_files
                rec_ind_2 = (i_batch_run+1)*n_batch_files
    
                data_subset = layer_dataset[rec_ind_1:rec_ind_2]
                
                # check if it is a convolutional layer.
                if conv:
                    
                    #Generation of cross surfaces, computed on multiple threads
                    results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                        (delayed(cross_tv_generator_conv)\
                                        (data_subset[recording], 
                                          self.n_input_channels,\
                                          n_input_features, cross_width,\
                                          self.taus)\
                                      for recording in range(len(data_subset)))
    
                    for i_result in range(len(results)):
                        batch_response=kmeans.predict(results[i_result])
                        cross_response.append([data_subset[i_result][0],\
                                               data_subset[i_result][1],\
                                                   batch_response])        
                
                else:
                                  
                    #Generation of cross surfaces, computed on multiple threads
                    results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                        (delayed(cross_tv_generator)\
                                        (data_subset[recording], 
                                          self.n_input_channels,\
                                          n_input_features,self.taus)\
                                      for recording in range(len(data_subset)))
                                            
                    for i_result in range(len(results)):
                        batch_response=kmeans.predict(results[i_result])
                        cross_response.append([data_subset[i_result][0],\
                                               batch_response])
                
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left "\
                          %(i_batch+1,total_batches,expected_t))                
                    batch_start_time = time.time()
            
    
        if self.verbose is True:    
            print("generatung time vectors took %s seconds." % (total_time))
            
        return cross_response 