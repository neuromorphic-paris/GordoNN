#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:42:52 2022

@author: marcorax93
"""

import numpy as np
import time

class GORDONN:
    """
    Gordon Network, managing Layers objects, and allowing save load states.
    
    Gordonn network arguments: 
    """
    def __init__(self, n_threads=8, verbose=False, server_mode=False):
        
        self.n_threads = n_threads
        self.verbose = verbose
        self.server_mode = server_mode
        self.layers = []
        self.architecture = []
        
    def add_layer(self, layer_type, layer_parameters):
        
        if layer_type=="Local": 
            self.architecture.append(layer_type)
        elif layer_type=="Cross":
            self.architecture.append(layer_type)
        elif layer_type=="Pool":
            self.architecture.append(layer_type)
        else:
            print("Please select one among these three classes: Local,Cross,Pool")

            
        
        