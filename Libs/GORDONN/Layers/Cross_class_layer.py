#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:03:43 2022

@author: marcorax93
"""

import numpy as np
import time
from joblib import Parallel, delayed 
from .Cross_Layer import Cross_Layer, \
                        cross_tv_generator,\
                        cross_tv_generator_conv

from tensorflow.keras.layers import Input, Dense, BatchNormalization,\
                                    Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers

class Cross_class_layer(Cross_Layer):
    """
    This is a modified Cross layer, that rather than clustering cross vectors,
    applies a shallow classifier like an mlp on the vector to act as an 
    event based classifier. This can only be selected as the last layer of the 
    network.
    
    Cross Layer constructor arguments: 
        
    n_hidden_units (int) : the number of hidden units of the mlp.
                       
    cross_tv_width (int): the width of the cross time vectors across channels,
                          polarity, if set to None the vector always spans across 
                          all the available channels, and it is not centered
                          on the reference event anymore. IF NUMBER
                          IT HAS TO BE ODD
    
    taus (float):  a list containing the time coefficient 
                   used for the local time vector creations 
                   for each channel of the cochlea, if it is
                   a single float, all channels/polarities
                   will have the same tau.
                   
    n_input_channels (int): the total number of channels or polarities of the 
                           previous layer.
                                                      
                   
    n_labels (int): number of labels in the dataset. 
    
    learning_rate (float): learning rate of the mlp.
    
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
    #TODO change the order here to better fit the layer
    def __init__(self, n_hidden_units, cross_tv_width, n_input_channels, 
                 taus, n_labels, learning_rate, mlp_ts_batch_size, mlp_epochs,
                 n_input_features=None, n_batch_files=None, dataset_runs=1,
                 n_threads=8, verbose=False):
        
        self.n_hidden_units = n_hidden_units
        self.cross_tv_width = cross_tv_width
        self.taus = taus
        self.n_input_channels = n_input_channels
        self.n_labels = n_labels
        self.learning_rate = learning_rate
        self.mlp_ts_batch_size = mlp_ts_batch_size
        self.mlp_epochs = mlp_epochs
        self.n_input_features = n_input_features
        self.n_batch_files = n_batch_files
        self.dataset_runs = dataset_runs
        self.n_threads = n_threads
        self.verbose = verbose
        
        # Output arguments for next layers
        self.n_output_features = n_features
        #Output channels is calculated only after training
        
    def learn(self, layer_dataset, labels):
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
        
        #input_size, hidden_size, output_size, learning_rate
        #TRAIN AN MLP HERE    
        mlp = create_mlp(input_size=n_input_features*cross_width, 
                            hidden_size=self.n_hidden_units,
                            output_size=self.n_labels, 
                            learning_rate=self.learning_rate)
                
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                rec_ind_1 = i_batch_run*n_batch_files
                rec_ind_2 = (i_batch_run+1)*n_batch_files

                data_subset = layer_dataset[rec_ind_1:rec_ind_2]
                labels_subset = labels[rec_ind_1:rec_ind_2]
                
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
          
                labels_per_ev = [labels_subset[recording]*np.ones(len(results[recording]), 
                                      dtype=int) for recording in range(len(data_subset))]
                
                # The final results of the local surfaces train dataset computation
                batch_local_tv = np.concatenate(results, axis=0)
                batch_local_labels = np.concatenate(labels_per_ev, axis=0, dtype=int)
                batch_local_labels_one_hot = np.zeros((batch_local_labels.size, 
                                                       batch_local_labels.max()+1),
                                                       dtype=int)
                batch_local_labels_one_hot[np.arange(batch_local_labels.size),batch_local_labels] = 1
                                
                if n_batches==1:
                    mlp.fit(batch_local_tv, batch_local_labels_one_hot,
                      epochs=self.mlp_epochs, batch_size=self.mlp_ts_batch_size, 
                      shuffle=True)
                else:
                    mlp.fit(batch_local_tv, batch_local_labels_one_hot,
                      epochs=self.mlp_epochs, batch_size=self.mlp_ts_batch_size, 
                      shuffle=True)
                
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
            
        # self.weights = mlp.weights
        self.mlp = mlp
        #TODO add a calculation for featues after removing zero pad
        if self.conv:
            self.n_output_channels = self.n_input_channels
        else:
            self.n_output_channels = None
            
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
        
        total_batches  = n_batches
        
        #Check if it is in convolutional mode or not
        if self.cross_tv_width == None or self.cross_tv_width >= self.n_input_channels :
            #Full layer
            cross_width = self.n_input_channels
            conv = False
        else:
            #Convolutional mode
            cross_width = self.cross_tv_width
            conv = True
        
            
        
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
                i_batch = i_batch_run                     
                expected_t = batch_time*(total_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left "\
                      %(i_batch+1,total_batches,expected_t))                
                batch_start_time = time.time()
            
    
        if self.verbose is True:    
            print("generatung time vectors took %s seconds." % (total_time))
            
        return cross_response 

    def tv_generation(self, layer_dataset, labels):
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
            
        n_files = len(layer_dataset)
        
        
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
            start_time = time.time()
        else:
            par_verbose = 0
        

       
        # check if it is a convolutional layer.
        if conv:
            
            # Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                (delayed(cross_tv_generator_conv)\
                                (layer_dataset[i_file], 
                                  self.n_input_channels,\
                                  n_input_features, cross_width,\
                                  self.taus)\
                              for i_file in range(n_files))
        
        else:
                          
            #Generation of cross surfaces, computed on multiple threads
            results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                (delayed(cross_tv_generator)\
                                (layer_dataset[i_file], 
                                  self.n_input_channels,\
                                  n_input_features,self.taus)\
                              for i_file in range(n_files))
  
        labels_per_ev = [labels[i_file]*np.ones(len(results[i_file]), 
                              dtype=int) for i_file in range(n_files)]
        
        # The final results of the local surfaces train dataset computation
        tv = np.concatenate(results, axis=0)
        labels_per_ev = np.concatenate(labels_per_ev, axis=0, dtype=int)
        labels_one_hot = np.zeros((labels_per_ev.size, 
                                         labels_per_ev.max()+1),
                                        dtype=int)
        labels_one_hot[np.arange(labels_per_ev.size),labels_per_ev] = 1
                                           
           

        if self.verbose is True:    
            total_time = start_time-time.time()
            print("generating time vectors took %s seconds." % (total_time))
            
        return tv, labels_one_hot
# =============================================================================
def create_mlp(input_size, hidden_size, output_size, learning_rate):
    """
    Function used to create a small mlp used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        mlp (keras model) : the freshly baked network
    """
    def relu_advanced(x):
        return K.activations.relu(x, alpha=0.3)
    
    inputs = Input(shape=(input_size,), name='encoder_input')
    x = BatchNormalization()(inputs)
    x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.3)(x)#0.3
    x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.7)(x)#0.7
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.9)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.5)(x)
    outputs = Dense(output_size, activation='sigmoid')(x)
    
    
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    mlp.compile(optimizer=adam,
              loss='categorical_crossentropy', metrics=['accuracy'])
    
    return mlp    