#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dic 2 13:51:42 2018

@author: marco, pedro

This file contains functions for generating time_context and time_surface on fly
Please note that these functions expect the dataset to be ordered from the 
lower timestamp to the highest!
"""
import numpy as np
from bisect import bisect_left


# =============================================================================
def Time_context(event_index, events, timecoeff, context_size):
    """
    Time_context is a function used to generate time_contexts starting from a 
    reference event and an array of events 
    Arguments : 
        event_index (int) : position of the reference event in the events list   
        events (list of events) : list of events (a recording) where each event is a 
                                  1D list containing in pos 0 the timestamp, in
                                  pos 1 the address and pos 2 for the rate if present
        timecoeff(float): exp decay coefficient for context generation
        context_size(int): the length of the context  
    Return :
        context (numpy array of floats) : monodimentional array of floats defining
                                          the time context for the reference event
    """ 
    context = np.zeros(context_size,dtype=float)
    context_size_counter = 1
    context[0] = 1 # The first value of each time context will always be 1
    if len(events)==3:
        context[0] = events[2][event_index]
    timestamp = events[0][event_index]
    address = events[1][event_index]
    ind = event_index-1 # Next index in the timestamps array to look for

    while (context_size_counter < context_size) and (ind>-1):
        if events[1][ind] == address:
            context[context_size_counter]=np.exp(-(timestamp-events[0][ind])/timecoeff)
            if len(events)==3:
                context[context_size_counter]*=events[2][ind]    
            context_size_counter += 1
        ind -= 1
     
    return context


# =============================================================================
def Time_Surface(xdim, ydim, event_index, timecoeff, dataset, minv=0.1):
    """
    Time_Surface_all: function that computes the Time_surface of an entire dataset,
    starting from a selected timestamp.
    Arguments : 
        ydim,xdim (int) : dimensions of the timesurface
        event_index (int) : position of the reference event in the events list   
        timecoeff (float) : the time coeff expressing the time decay
        dataset (nested lists) : dataset containing the events, it might have rates
        num_polarities (int) : total number of labels or polarities of the time surface 
        minv (float) : hardtreshold for the time surface, values smaller than minv will be
               removed from the result 
      
   Returns :
       tsurface (numpy array of floats) : time surface of length num_polarities*xdim*ydim 
    """
    if len(dataset)==4:
        tmpdata = [dataset[0], dataset[1], dataset[2], dataset[3]]
    else:
        tmpdata = [dataset[0], dataset[1], dataset[2]]
    timestamp = dataset[0][event_index]
    #taking only the timestamps before the reference 
    ind_subset = np.concatenate((np.ones(event_index,bool), np.zeros(len(tmpdata[0])-(event_index),bool)))
    if len(dataset)==4:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset], tmpdata[3][ind_subset]]
    else:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]
    #removing all the timestamps that will generate values below minv
    min_timestamp = timestamp + timecoeff*np.log(minv) #timestamps<min_timestamp WILL BE DISCARDED 
    ind = bisect_left(tmpdata[0],min_timestamp)
    ind_subset = np.concatenate((np.zeros(ind,bool), np.ones(len(tmpdata[0])-(ind),bool)))
    if len(dataset)==4:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset], tmpdata[3][ind_subset]]
        tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)*tmpdata[3]
    else:
        tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]
        tsurface_array = np.exp((tmpdata[0]-timestamp)/timecoeff)
    #now i need to build a matrix that will represents my surface, i will take 
    #only the highest value for each x and y as the other ar less informative
    #and we want each layer be dependant on the timecoeff of the timesurface
    #Note that exp is monotone and the timestamps are ordered, thus the last 
    #values of the dataset will be the lowest too
    tsurface = np.zeros([ydim,xdim])
    for i in range(len(tsurface_array)):
        tsurface[tmpdata[1][i],tmpdata[2][i]]=tsurface_array[i]
    del tmpdata,timestamp, tsurface_array
    
    return tsurface.flatten()