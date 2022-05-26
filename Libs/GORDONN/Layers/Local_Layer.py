#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:03:43 2022

@author: marcorax93
"""

import numpy as np
import time
from joblib import Parallel, delayed 
from sklearn.cluster import  MiniBatchKMeans
import matplotlib.pyplot as plt

class Local_Layer:
    """
    Local Layer for GORDONN architecture, used to detect temporal patterns in 
    single channels of the neuromorphic cochlea. It is meant to be the first 
    layer of the network.
    
    Local Layer constructor arguments: 
        
    n_features (int) : the number of features or centers extracted by the 
                       local layer.
                       
    local_tv_length (int): the length of the local time vectors.
    
    taus (list of floats or float):  a list containing the time coefficient 
                                       used for the local time vector creations 
                                       for each channel of the cochlea, if it is
                                       a single float, all channels/polarities
                                       will have the same tau.
    
    n_input_channels (int): the total number of channels or polarities of the 
                           previous layer.
                                                      
    
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
    def __init__(self, n_features, local_tv_length, taus, n_input_channels, 
                 n_input_features = None, n_batch_files=None, dataset_runs=1,
                 n_threads=8, verbose=False):
        
        self.n_features = n_features
        self.local_tv_length = local_tv_length
        self.taus = taus
        self.n_input_channels = n_input_channels
        self.n_input_features = n_input_features
        self.n_batch_files = n_batch_files
        self.dataset_runs = dataset_runs
        self.n_threads = n_threads
        self.verbose = verbose
        
        # Output arguments for next layers
        self.n_output_channels = n_input_channels
        self.n_output_features = n_features


        
    def learn(self, layer_dataset, labels):
        """
        Method to process and learn local features.
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        
        Arguments: 
             layer_dataset: list of individual event based recoriding as
                            generated by the cochlea
            
             labels: list of labels index of each recording of layer_dataset
            
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
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' LOCAL TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
          
        kmeans = MiniBatchKMeans(n_clusters=self.n_features,
                                 verbose=par_verbose)
        
        kmeans._n_threads = self.n_threads
        
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                rec_ind_1 = i_batch_run*n_batch_files
                rec_ind_2 = (i_batch_run+1)*n_batch_files

                data_subset = layer_dataset[rec_ind_1:rec_ind_2]
                
                # Generation of local surfaces, computed on multiple threads
                results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                    (delayed(local_tv_generator)\
                                    (data_subset[recording],\
                                     self.n_input_channels,\
                                     taus, self.local_tv_length)\
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
            
    def predict(self, layer_dataset, labels):
        """
        Method to generate local features response
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        Arguments: 
             layer_dataset: list of individual event based recoriding as
                            generated by the cochlea
                            
             labels: list of labels index of each recording of layer_dataset               
        
        Returns: 
            local_response: list of individual event based recordings with a
                            new array of the corresponfing feature index
                            per each event
            
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
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
        
        total_batches  = n_batches
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' LOCAL TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
          
        kmeans = MiniBatchKMeans(n_clusters=self.n_features,
                                 verbose=par_verbose)
        
        kmeans._n_threads = self.n_threads
        kmeans.cluster_centers_ = self.features
        
        local_response=[]
        for i_batch_run in range(n_batches):
            
            rec_ind_1 = i_batch_run*n_batch_files
            rec_ind_2 = (i_batch_run+1)*n_batch_files

            data_subset = layer_dataset[rec_ind_1:rec_ind_2]
            
            # Generation of local vectors, computed on multiple threads
            results = Parallel(n_jobs=self.n_threads, verbose=par_verbose)\
                                (delayed(local_tv_generator)\
                                (data_subset[recording],\
                                 self.n_input_channels,\
                                 taus, self.local_tv_length)\
                                 for recording in range(len(data_subset)))     
        
            
            for i_result in range(len(results)):
                batch_response=kmeans.predict(results[i_result])
                local_response.append([data_subset[i_result][0],\
                                       data_subset[i_result][1], batch_response])
            
            if self.verbose is True: 
                batch_time = time.time()-batch_start_time
                i_batch = i_batch_run                     
                expected_t = batch_time*(total_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left "\
                      %(i_batch+1,total_batches,expected_t))                
                batch_start_time = time.time()
        

        if self.verbose is True:    
            print("generating time vectors took %s seconds." % (total_time))
            
        return local_response
    
    
    #Importing Classifiers Methods
    from Libs.GORDONN.Classifiers.Histogram_Classifiers import gen_histograms
    from Libs.GORDONN.Classifiers.Histogram_Classifiers import gen_signatures
            
    def response_plot(self, local_response, f_index, class_name = None):
        """
        Function used to generate plots of local layer response.
        It Plots the input data first, a heatmap of the sum(local tv)
        to show amplitude information
        And finally the scatter plot of events colored by feature index.
        """
        # Build the vector of individual taus
        if type(self.taus) is np.ndarray or type(self.taus) is list :
            taus = np.array(self.taus)
        else:
            taus = self.taus*np.ones(self.n_input_channels)
        
        timestamps = local_response[f_index][0]
        channels =  local_response[f_index][1]
        features = local_response[f_index][2]
        
        # First print the original recording
        plt.figure()
        plt.suptitle('Original file: '+ str(f_index) +' Class: '+ str(class_name), fontsize=16)
        plt.scatter(timestamps, channels, s=1)
        xlims = plt.xlim()
        ylims = plt.ylim()
        lcs = local_tv_generator(local_response[f_index],\
                                 self.n_input_channels,\
                                 taus,\
                                 self.local_tv_length)
        # Plot heatmap
        plt.figure()
        plt.suptitle('Heatmap file: '+ str(f_index) +' Class: '+ str(class_name), fontsize=16)
        image=plt.scatter(timestamps, channels,
                          c=np.sum(lcs,1), s=0.1)
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.colorbar(image)     
        
        plt.figure()
        for i_feature in range(self.n_features):
            # Extract events by feature
            indx = features==i_feature

            image = plt.scatter(timestamps[indx], channels[indx],\
                                label='Feature '+str(i_feature))
                
                
    def features_plot(self):
        """
        Function used to generate a plot of the local layer features.
        """
        
        plt.figure()
        plt.plot(np.transpose(self.features))
        
def local_tv_generator(recording_data, n_polarities, taus, context_length):
    """
    Function used to generate local time vectors.
    
    Arguments:
        
        recording_data: (list of 2 numpy 1D arrays) It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities: (int) the number of channels/polarities of the dataset.
        taus: (list of floats) the decays of the local time vector layer per each polarity.
        context_length: (int) the length of the local time vector 
                              (the length of its context).  
    
    Returns:
        
        local_tvs: (2D numpy array of floats) the timevectors generated for the 
                   recording where the dimensions are [i_event, timevector element]
    """
    
    n_events = len(recording_data[0])
    local_tvs=np.zeros([n_events,context_length])
    for polarity in range(n_polarities):
        indxs_channel = np.where(recording_data[1]==polarity)[0]
        new_local_tv = np.zeros(context_length)
        for i,ind in enumerate(indxs_channel):
            timestamp = recording_data[0][ind]
            # timestamps = [recording_data[0][j] for j in indxs_channel[(i+1-context_length):i][::-1]]
            context = recording_data[0][indxs_channel[(i+1-context_length):i][::-1]]
            new_local_tv[0] = 1         
            exponent=-(timestamp-context)/taus[polarity]
            exponent[exponent<-10] = -10 #To avoid overflow 
            new_local_tv[1:1+len(context)] = np.exp(exponent)
            local_tvs[ind] = new_local_tv

    return local_tvs        

