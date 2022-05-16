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


class Cross_Layer:
    """
    Cross Layer for GORDONN architecture, used to detect temporal patterns in 
    single channels of the neuromorphic cochlea. It can also be used as an 
    initial layer with specific taus per channel
    
    Cross Layer constructor arguments: 
        
    n_features (int) : the number of features or centers extracted by the 
                       local layer.
                       
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
    def __init__(self, n_features, cross_tv_width, n_input_channels, 
                 taus, n_input_features=None, n_batch_files=None,
                 dataset_runs=1, n_threads=8, verbose=False):
        
        self.n_features = n_features
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
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
        
        
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
                i_batch = i_batch_run + n_batches                    
                expected_t = batch_time*(total_batches-i_batch-1)
                total_time += (time.time() - batch_start_time)
                print("Batch %i out of %i processed, %s seconds left "\
                      %(i_batch+1,total_batches,expected_t))                
                batch_start_time = time.time()
            
    
        if self.verbose is True:    
            print("generatung time vectors took %s seconds." % (total_time))
            
        return cross_response 
        
    def gen_histograms(self, cross_response):
        """
        Function used to generate histograms of cross layer response.
        """
        n_recordings = len(cross_response)
        hists = np.zeros([n_recordings,self.n_input_channels, self.n_features])
        norm_hists = np.zeros([n_recordings,self.n_input_channels, self.n_features])
        for recording_i,data in enumerate(cross_response):
            data = data[1:]
            indx, occurences = np.unique(data, axis=1, return_counts=True)
            indx = np.asarray(indx, dtype=(int))
            hists[recording_i,indx[0],indx[1]] = occurences
            norm_hists[recording_i,indx[0],indx[1]] = occurences/sum(occurences)
        
        return hists, norm_hists
    
    def gen_signatures(self, histograms, norm_histograms, classes, labels):
        """
        Function used to generate signatures of cross layer response.
        Signatures are average histograms of recording for every class.
        """
        n_labels = len(classes)
        signatures = np.zeros([n_labels, self.n_input_channels, self.n_features])
        norm_signatures = np.zeros([n_labels, self.n_input_channels, self.n_features])
        for class_i in range(n_labels):
            indx = labels==class_i
            signatures[class_i] = np.mean(histograms[indx], axis=0)
            norm_signatures[class_i] = np.mean(norm_histograms[indx], axis=0)
            
        return signatures, norm_signatures        

    #TODO obtain the original channel of the event
    def response_plot(self, cross_response, f_index, class_name = None):
        """
        Function used to generate plots of cross layer response.
        It Plots the input data first, and the scatter plot of events 
        colored by feature index.
        """
        
        timestamps = cross_response[f_index][0]
        channels =  cross_response[f_index][1]
        features = cross_response[f_index][2]
        
        # First print the original recording
        plt.figure()
        plt.suptitle('Original file: '+ str(f_index) +' Class: '+ str(class_name), fontsize=16)
        plt.scatter(timestamps, channels, s=1)
   
        plt.figure()
        for i_feature in range(self.n_features):
            # Extract events by feature
            indx = features==i_feature
    
            image = plt.scatter(timestamps[indx], channels[indx],\
                                label='Feature '+str(i_feature))

def cross_tv_generator(recording_data, n_polarities, features_number, taus):
    """
    Function used to generate cross time vectors.
    
    Arguments:
        
        recording_data (list of 2 numpy 1D arrays): It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities (int): the number of channels/polarities of the dataset.
        features_number (int): the number of features of the previous layer
        taus (array of floats): the single decays of the cross time vector layer
                               per channel.
        context_length (int): the length of the cross time vector 
                              (the length of its context).  
    
    Returns:
        
        cross_tvs (2D numpy array of floats): the timevectors generated for the 
                   recording where the dimensions are [i_event, flattened 
                                                       2D timesurface element]
    """
    
    
    n_events = len(recording_data[0])
    
    # 2D timesurface dimension
    ydim,xdim = [n_polarities, features_number]
    cross_tvs = np.zeros([len(recording_data[0]),xdim*ydim], dtype="float16")
    timestamps = np.zeros([ydim, xdim], dtype=int)
    
    # create a taus matrix with the tiling taus to the size of timestamp
    taus_m = np.transpose(np.tile(taus,[xdim,1]))
    
    if features_number>1:
        features = recording_data[2]
    else:
        features = np.zeros(n_events, dtype=int)
        
    for event_ind in range(n_events):   
        new_timestamp = recording_data[0][event_ind]                                   
        polarity = recording_data[1][event_ind]
        feature = features[event_ind]
        timestamps[polarity, feature] = recording_data[0][event_ind]
        timesurf = np.exp(-(new_timestamp-timestamps)/taus_m)*(timestamps!=0)   
        cross_tvs[event_ind] = (timesurf).flatten() 


    return cross_tvs


    

                
def cross_tv_generator_conv(recording_data, n_polarities, features_number, 
                            cross_width, taus):
    """
    Function used to generate cross time vectors.
    
    Arguments:
        
        recording_data: (list of 2 numpy 1D arrays) It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities: (int) the number of channels/polarities of the dataset.
        features_number (int): the number of features of the previous layer
        cross_width (int): the lateral size of the cross time vector.
        taus (array of floats): the single decays of the cross time vector layer
                               per channel/polarity. 
    
    Returns:
        
        cross_tvs (2D numpy array of floats): the timevectors generated for the 
                   recording where the dimensions are [i_event, flattened 
                                                       2D timevector element]
    """
    
    
    n_events = len(recording_data[0])
    
    # 2D timesurface dimension
    ydim,xdim = [n_polarities, features_number]
    cross_tvs_conv = np.zeros([len(recording_data[0]),cross_width*xdim], dtype="float16")
    zerp_off =  cross_width//2#zeropad offset
    timestamps = np.zeros([ydim+zerp_off*2, xdim], dtype=int) # ydim + 2* zeropad
    
    # create a taus matrix with the tiling taus to the size of timestamp
    taus_m = np.ones([ydim+zerp_off*2])
    taus_m[zerp_off:-zerp_off] = taus
    taus_m = np.transpose(np.tile(taus_m,[xdim,1]))
    
    if features_number>1:
        features = recording_data[2]
    else:
        features = np.zeros(n_events, dtype=int)
    
    local_ind = np.arange(cross_width)    
    for event_ind in range(n_events):   
        new_timestamp = recording_data[0][event_ind]                                   
        polarity = recording_data[1][event_ind]
        feature = features[event_ind]
        timestamps[polarity+zerp_off, feature] = recording_data[0][event_ind]
        exponent = -(new_timestamp-timestamps)/taus_m
        exponent[exponent<-10] = -10 #To avoid overflow 
        exponent[exponent>0] = 0 #To avoid positive exponentials 
        timesurf = np.exp(exponent)*(timestamps!=0)   
        timesurf = timesurf[local_ind+polarity]
        cross_tvs_conv[event_ind,:] = (timesurf).flatten() 


    return cross_tvs_conv

