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
from bisect import bisect_left, bisect_right

# Time_context is a function used to generate time_contexts starting from a 
# reference event and an array of events 
# =============================================================================
# Args : 
#       event_index(int): position of the reference event in the events list   
#       events(list of event): list of events (a batch) where each event is a 
#                              1D list containing in pos 0 the timestamp and in
#                              pos 1 the address 
#       timecoeff(float): exp decay coefficient for context generation
#       context_size(int): the length of the context                
# =============================================================================
def Time_context(event_index, events, timecoeff, context_size):
    context = np.zeros(8,dtype=float)
    context_size_counter = 1
    context[0] = 1 # The first value of each time context will always be 1
    timestamp = events[0][event_index]
    address = events[1][event_index]
    ind = event_index-1 # Next index in the timestamps array to look for

    while (context_size_counter < context_size) and (ind>-1):
        if events[1][ind] == address:
            context[context_size_counter]=(np.exp(-(timestamp-events[0][ind])/timecoeff))
            context_size_counter += 1
        ind -= 1
     
    return np.array(context)

## Time_Surface: function that computes the Time_Surface starting from a 
# reference event and an array of events 
# =============================================================================
# ydim,xdim : dimensions of the timesurface (please note that 
#             they are encoded as monodimensional arrays)
# event_index(int): position of the reference event in the events list   
# timecoeff : the time coeff expressing the time decay
# dataset : dataset containing the events, a list where dataset[0] contains the 
#           timestamps as microseconds, and dataset[1] contains [x,y] pixel positions 
# num_polarities : total number of labels or polarities of the time surface 
# minv : hardtreshold for the time surface, values smaller than minv will be
#        removed from the result 
#
# tsurface : matrix of size num_polarities*xdim*ydim 
# =============================================================================
def Time_Surface(xdim, ydim, event_index, timecoeff, dataset, minv=0.1):
    tmpdata = [dataset[0], dataset[1], dataset[2]]
    timestamp = dataset[0][event_index]
    #taking only the timestamps before the reference 
    ind_subset = np.concatenate((np.ones(event_index,bool), np.zeros(len(tmpdata[0])-(event_index),bool)))
    tmpdata = [tmpdata[0][ind_subset], tmpdata[1][ind_subset], tmpdata[2][ind_subset]]    
    #removing all the timestamps that will generate values below minv
    min_timestamp = timestamp + timecoeff*np.log(minv) #timestamps<min_timestamp WILL BE DISCARDED 
    ind = bisect_left(tmpdata[0],min_timestamp)
    ind_subset = np.concatenate((np.zeros(ind,bool), np.ones(len(tmpdata[0])-(ind),bool)))
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