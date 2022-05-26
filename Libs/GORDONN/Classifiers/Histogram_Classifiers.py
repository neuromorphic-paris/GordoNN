#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:04:51 2022

@author: marcorax93

Library containing methods and functions for histogram-of-activity based 
classifiers
"""


import numpy as np
from sklearn import svm

###Functions###
def hist_classifier(histograms,signatures,labels):
    """
    Simple classifier based on euclidean distance between histograms and signatures
    """
    
    accuracy=0
    n_labels = len(labels)
    
    #Recordings in which the network didn't respond (MI layers likely cut out
    #too many events)
    zero_hist_indx = np.sum(histograms,axis=(1,2))!=0
    
    for ind,label in enumerate(labels):
        if zero_hist_indx[ind]:
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
    
    #Recordings in which the network didn't respond (MI layers likely cut out
    #too many events)
    train_zero_hist_indx = np.sum(train_histograms,axis=(1,2))!=0
    test_zero_hist_indx = np.sum(test_histograms,axis=(1,2))!=0

    
    flat_train_hist = train_histograms.reshape(train_histograms.shape[0],\
                        train_histograms.shape[1]*train_histograms.shape[2]) 
        
    flat_test_hist = test_histograms.reshape(test_histograms.shape[0],\
                        test_histograms.shape[1]*test_histograms.shape[2])   
    SVC = svm.SVC(decision_function_shape='ovo')
    SVC.fit(flat_train_hist[train_zero_hist_indx], train_labels[train_zero_hist_indx])
    
    pred_labels = SVC.predict(flat_test_hist)
    n_correct_labels = sum((pred_labels[test_zero_hist_indx]-test_labels[test_zero_hist_indx])==0)
    accuracy=n_correct_labels*100/len(test_labels) 
    return accuracy    

###METHODS###
def gen_histograms(self, neuro_response):
    """
    Function used to generate histograms of a neuromorphic layer response.
    """
    n_recordings = len(neuro_response)
    
    if self.n_output_channels==None:
        n_output_channels=1
    else:
        n_output_channels=self.n_output_channels
        
    if self.n_output_features==None:
        n_output_features=1
    else:
        n_output_features=self.n_output_features
    
    hists = np.zeros([n_recordings, n_output_channels, n_output_features])
    norm_hists = np.zeros([n_recordings, n_output_channels, n_output_features])
    for recording_i,data in enumerate(neuro_response):
        if len(data[0]):
            data = data[1:] #discarding the timestamp information
            indx, occurences = np.unique(data, axis=1, return_counts=True)
            indx = np.asarray(indx, dtype=(int))
            if self.n_output_channels==None:
                hists[recording_i,0,indx[0]] = occurences
                norm_hists[recording_i,0,indx[0]] = occurences/sum(occurences)
            elif self.n_output_features==None:
                hists[recording_i,indx[0],0] = occurences
                norm_hists[recording_i,indx[0],0] = occurences/sum(occurences)
            else:
                hists[recording_i,indx[0],indx[1]] = occurences
                norm_hists[recording_i,indx[0],indx[1]] = occurences/sum(occurences)
            

    return hists, norm_hists

def gen_signatures(self, histograms, norm_histograms, classes, labels):
    """
    Function used to generate signatures of neuromorphic layers response.
    Signatures are average histograms of recording for every class.
    """
    n_labels = len(classes)
    
    if self.n_output_channels==None:
        n_output_channels=1
    else:
        n_output_channels=self.n_output_channels
        
    if self.n_output_features==None:
        n_output_features=1
    else:
        n_output_features=self.n_output_features
    
    signatures = np.zeros([n_labels, n_output_channels, n_output_features])
    norm_signatures = np.zeros([n_labels, n_output_channels, n_output_features])
    zero_hist_indx = np.sum(histograms,axis=(1,2))!=0
       
    for class_i in range(n_labels):
        indx = labels==class_i        
        indx = indx*zero_hist_indx
        signatures[class_i] = np.mean(histograms[indx], axis=0)
        norm_signatures[class_i] = np.mean(norm_histograms[indx], axis=0)
        
    return signatures, norm_signatures