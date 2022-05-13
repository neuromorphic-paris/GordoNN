#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:42:52 2022

@author: marcorax93
"""

import numpy as np
import time,copy
from sklearn import svm

from .Layers.Local_Layer import Local_Layer
from .Layers.Cross_Layer import Cross_Layer
from .Layers.Pool_Layer import Pool_Layer

class GORDONN:
    """
    Gordon Network, managing Layers objects, and allowing save load states.
    
    Gordonn network arguments (can be acessed as attributes with the same name
                               ex: NetObj.n_threads): 
        n_threads (int): Number of threads used for computing and learning Time
                         Vectors
        verbose (boolean): Verbose mode of the network. Useful to keep track of 
                           progress.
        server_mode (boolean): If True the network will save verbose terminal 
                               messages on a log, useful to keep track of progress
                               in detached terminals
        low_memory_mode (boolean): If False the network will save the output of 
                                   every layer, useful for testing, but memory
                                   intensive for deep layered networks
    """
    def __init__(self, n_threads=8, verbose=False, server_mode=False, low_memory_mode=True):
        
        self.n_threads = n_threads
        self.verbose = verbose
        self.server_mode = server_mode
        self.low_memory_mode = low_memory_mode
        self.layers = []
        self.architecture = []
        
        
    def add_layer(self, layer_type, layer_parameters):
        
        if layer_type=="Local": 
            
            self.architecture.append(layer_type)
            
            [n_features, local_tv_length, n_input_channels, taus,\
                n_batch_files, dataset_runs] = layer_parameters
                
            self.layers.append(Local_Layer(n_features, local_tv_length,\
                                           n_input_channels, taus, \
                                           n_batch_files, dataset_runs, 
                                           self.n_threads, self.verbose))
                
        elif layer_type=="Cross":
            self.architecture.append(layer_type)
            
            [n_features, cross_tv_width, 
                                n_input_channels, taus, 
                                n_input_features, n_batch_files,
                                dataset_runs] = layer_parameters
                
            self.layers.append(Cross_Layer(n_features, cross_tv_width,\
                                           n_input_channels, taus, n_input_features,\
                                           n_batch_files, dataset_runs, 
                                           self.n_threads, self.verbose))
        elif layer_type=="Pool":
            self.architecture.append(layer_type)
            [n_input_channels, pool_factor] = layer_parameters
            self.layers.append(Pool_Layer(n_input_channels, pool_factor))
            
        else:
            print("Please select one among these three classes: Local,Cross,Pool")

            
        
    def learn(self, dataset_train, labels_train, classes, rerun_layer=0):
        """
        Network learning method. It saves net responses recallable with net_response_train.
        
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner

            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset = dataset_train
        n_layers = len(self.layers)

        #check if you only want to rerun a layer
        if rerun_layer == 0:
            layers_index = np.arange(n_layers)
            self.net_response_train=[]
            self.layer_dataset_train = []
        else:
            layers_index = np.arange(rerun_layer,n_layers)
            layer_dataset = self.layer_dataset_train[rerun_layer]
            self.layer_dataset_train = self.layer_dataset_train[:rerun_layer]
            self.net_response_train = self.net_response_train[:rerun_layer]
        
        #Run the network
        for layer_i in layers_index:
            
            #Empty last layer memory if memory mode
            if layer_i!=0 and self.low_memory_mode:
                self.layer_dataset_train[layer_i-1]=[]
                self.net_response_train[layer_i-1]=[]

            
            self.layer_dataset_train.append(layer_dataset)
            if self.architecture[layer_i]=="Pool":
                layer_dataset=self.layers[layer_i].pool(layer_dataset)
                self.net_response_train.append(layer_dataset)
            else:
                self.layers[layer_i].learn(layer_dataset)
                layer_dataset=self.layers[layer_i].predict(layer_dataset)
                self.net_response_train.append(layer_dataset)
                hists, norm_hists = self.layers[layer_i].gen_histograms(layer_dataset)
                sign, norm_sign = self.layers[layer_i].gen_signatures(hists, norm_hists, classes, labels_train)
                self.layers[layer_i].hists = hists
                self.layers[layer_i].norm_hists = norm_hists
                self.layers[layer_i].sign = sign
                self.layers[layer_i].norm_sign = norm_sign
                
    def predict(self, dataset_test, labels_train, labels_test, classes, rerun_layer=0):
        """
        Network learning method. It saves net responses recallable with net_response_test.
        
        Arguments:
            dataset (nested lists) : the dataset used to extract features in a unsupervised 
                                     manner

            rerun_layer (int) : If you want to rerun a layer (remember that the
                                net is sequential, if you run a layer then the 
                                net WILL HAVE TO run all the layers on top of the 
                                one selected), if rerun_layer is 2 for a 4 layer
                                net, then the network will run the 2,3,4 layer
                                keeping only layer 0,1 
        """
        
        layer_dataset = dataset_test
        n_layers = len(self.layers)

        #check if you only want to rerun a layer
        if rerun_layer == 0:
            layers_index = np.arange(n_layers)
            self.net_response_test=[]
            self.layer_dataset_test = []
        else:
            layers_index = np.arange(rerun_layer,n_layers)
            layer_dataset = self.layer_dataset_test[rerun_layer]
            self.layer_dataset_test = self.layer_dataset_test[:rerun_layer]
            self.net_response_test = self.net_response_test[:rerun_layer]
        
        #Run the network
        for layer_i in layers_index:

            #Empty last layer memory if memory mode
            if layer_i!=0 and self.low_memory_mode:
                self.layer_dataset_test[layer_i-1]=[]
                self.net_response_test[layer_i-1]=[]
                
                
            self.layer_dataset_test.append(layer_dataset)
            if self.architecture[layer_i]=="Pool":
                layer_dataset=self.layers[layer_i].pool(layer_dataset)
                self.net_response_test.append(layer_dataset)
            else:
                layer_dataset=self.layers[layer_i].predict(layer_dataset)
                self.net_response_test.append(layer_dataset)
                hists, norm_hists = self.layers[layer_i].gen_histograms(layer_dataset)
                self.layers[layer_i].hists_test = hists
                self.layers[layer_i].norm_hists_test = norm_hists
                self.layers[layer_i].hist_accuracy = hist_classifier(hists,\
                                                                     self.layers[layer_i].sign,\
                                                                     labels_test)
                self.layers[layer_i].norm_hist_accuracy = hist_classifier(norm_hists,\
                                                                     self.layers[layer_i].norm_sign,\
                                                                     labels_test)
                    
                self.layers[layer_i].svm_hist_accuracy = svm_hist_classifier(self.layers[layer_i].hists,\
                                                                             hists,\
                                                                             labels_train,\
                                                                             labels_test)
                    
                self.layers[layer_i].svm_norm_hist_accuracy = svm_hist_classifier(self.layers[layer_i].norm_hists,\
                                                                             norm_hists,\
                                                                             labels_train,\
                                                                             labels_test)
            
#TODO should move the classifiers somewhere else                
def hist_classifier(histograms,signatures,labels):
    """
    Simple classifier based on euclidean distance between histograms and signatures
    """
    
    accuracy=0
    n_labels = len(labels)
    for ind,label in enumerate(labels):
        hist = histograms[ind]
        pred_label = np.argmin(np.sum((hist-signatures)**2, axis=(1,2)))
        if pred_label==label:
            accuracy += 1
    accuracy=accuracy*100/n_labels 
    return accuracy
    
def svm_hist_classifier(train_histograms, test_histograms, train_labels,\
                        test_labels):
    """
    Simple classifier based on euclidean distance between histograms and signatures
    """
    
    flat_train_hist = train_histograms.reshape(train_histograms.shape[0],\
                        train_histograms.shape[1]*train_histograms.shape[2]) 
        
    flat_test_hist = test_histograms.reshape(test_histograms.shape[0],\
                        test_histograms.shape[1]*test_histograms.shape[2])   
    SVC = svm.SVC(decision_function_shape='ovo')
    SVC.fit(flat_train_hist, train_labels)
    pred_labels = SVC.predict(flat_test_hist)
    n_correct_labels = sum((pred_labels-test_labels)==0)
    accuracy=n_correct_labels*100/len(test_labels) 
    return accuracy    