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
def Time_context_mod(event_index, events, timecoeff, context_size, last_contexts, padding_counter, cross_correlation_th, auto_variance):
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
        last_contexts (list of arrays) : The last contexts produced so far, divided per polarity
                                    useful to remove redundant contexts

        cross_correlation_th (float) : The Mean Squared difference threshold between 
                                           two close contexts under which the latter is 
                                           removed
        auto_variance (float) : The Variance threshold under which the context 
                                produced is removed
    Return :
        context (numpy array of floats) : monodimentional array of floats defining
                                          the time context for the reference event
    """ 
    
    context = np.zeros(context_size, dtype="float16")
    context_size_counter = 1
    context[0] = 1 # The first value of each time context will always be 1
    if len(events)==3:
        context[0] = events[2][event_index]
    timestamp = events[0][event_index]
    address = events[1][event_index]
    ind = event_index-1 # Next index in the timestamps array to look for
    
    #trash the first events (padding) depending on context length
    padding_counter[address]+=1
    if padding_counter[address] < context_size:
        return []
    
    while (context_size_counter < context_size) and (ind>-1):
        if events[1][ind] == address:
            context[context_size_counter]=np.exp(-(timestamp-events[0][ind])/timecoeff)
            if len(events)==3:
                context[context_size_counter]*=events[2][ind]    
            context_size_counter += 1
        ind -= 1
    
    context_mean = np.sum(context)/context.size
    context_variance = np.sum((context-context_mean)**2)/context.size
    if context_variance <= auto_variance:
        return []
    
    if last_contexts[address].size:
#        cross_context_correlation = np.correlate(context, last_contexts[address])[0]
#        auto_context_correlation = np.correlate(context, context)[0]
        cross_context_correlation = np.sum(context)
        auto_context_correlation = np.sum(last_contexts[address])      
        if np.abs(cross_context_correlation-auto_context_correlation) <= cross_correlation_th:
            return []
    
    last_contexts[address] = context
    
    return context

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
    context = np.zeros(context_size,dtype="float16")
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
def Time_context_later_mod(event_index, events, timecoeff, context_size,  last_contexts,  cross_correlation_th, auto_variance):
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
    naddress = len(events[1][0])
    
    # in case there are not enough events to completely load the timesurface
    if event_index<(context_size-1):
        # contexts = np.zeros([naddress,context_size],dtype=float)
        # contexts_size_counter = 1
        # contexts[:,0] = events[1][event_index]
        # timestamp = events[0][event_index]      
        # ind = event_index-1 # Next index in the timestamps array to look for
        # while (contexts_size_counter < context_size) and (ind>-1):
        #     contexts[:,contexts_size_counter]=np.exp(-(timestamp-events[0][ind])/timecoeff)
        #     contexts[:,contexts_size_counter]*=events[1][ind]    
        #     contexts_size_counter += 1
        #     ind -= 1
        return [[] for i in range(naddress)]
    else:
        timestamp = events[0][event_index] 
        contexts = events[1][event_index:event_index-context_size:-1].transpose()*np.exp(-(timestamp-events[0][event_index:event_index-context_size:-1])/timecoeff)
    
    result=[]
    
    for address in range(naddress):
        context=contexts[address]
        context_mean = np.sum(context)/context.size
        context_variance = np.sum((context-context_mean)**2)/context.size
        if context_variance <= auto_variance:
            result.append([])   
        elif last_contexts[address].size:
    #        cross_context_correlation = np.correlate(context, last_contexts[address])[0]
    #        auto_context_correlation = np.correlate(context, context)[0]
            cross_context_correlation = np.sum(context[0])
            auto_context_correlation = np.sum(last_contexts[address][0])     
            if np.abs(cross_context_correlation-auto_context_correlation) <= cross_correlation_th:
                result.append([])   
            else:
                last_contexts[address] = context
                result.append(context)
        else:
            last_contexts[address] = context
            result.append(context)
    
    return result

# =============================================================================
def Time_context_later(event_index, events, timecoeff, context_size):
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
    # in case there are not enough events to completely load the dimesurface
    if event_index<=context_size-1:
        contexts = np.zeros([len(events[1][0]),context_size],dtype=float)
        context_size_counter = 1
        contexts[:,0] = events[1][event_index]
        timestamp = events[0][event_index]      
        ind = event_index-1 # Next index in the timestamps array to look for
        while (context_size_counter < context_size) and (ind>-1):
            contexts[:,context_size_counter]=np.exp(-(timestamp-events[0][ind])/timecoeff)
            contexts[:,context_size_counter]*=events[1][ind]    
            context_size_counter += 1
            ind -= 1
    else:
        timestamp = events[0][event_index] 
        contexts = events[1][event_index:event_index-context_size:-1].transpose()*np.exp(-(timestamp-events[0][event_index:event_index-context_size:-1])/timecoeff)
        
    return contexts

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
    tsurface = np.zeros([ydim,xdim], dtype="float16")
    for i in range(len(tsurface_array)):
        tsurface[tmpdata[1][i],tmpdata[2][i]]=tsurface_array[i]
    del tmpdata,timestamp, tsurface_array
    
    return tsurface.flatten()