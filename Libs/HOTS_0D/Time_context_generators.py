#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dic 2 13:51:42 2018

@author: marco, pedro

This file contains functions for generating time_context on fly
Please note that these functions expect the dataset to be ordered from the 
lower timestamp to the highest
"""
import numpy as np

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
    context = []
    context_size_counter = 1
    context.append(1.) # The first value of each time context will always be 1
    timestamp = events[0][event_index]
    address = events[1][event_index]
    ind = event_index-1 # Next index in the timestamps array to look for

    while (context_size_counter < context_size) and (ind>-1):
        if events[1][ind] == address:
            context.append(np.exp(-(timestamp-events[0][ind])/timecoeff))
            context_size_counter += 1
        ind -= 1
    
    for i in range(context_size-context_size_counter):
        context.append(0.)
    return np.array(context)
