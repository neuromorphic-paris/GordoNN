# -*- coding: utf-8 -*-
"""
Created on Mar Jan 15 11:36:42 2019

@author: marcorax, pedro

This file contains functions to automatically loads and shuffles datasets
 
"""
import numpy as np
import time
from Libs.Data_loading.get_filenames_dataset import get_filenames_on_off_dataset
from Libs.Data_loading.AERDATA_load import AERDATA_load

# A function used to load the ON OFF dataset
# =============================================================================
#    Args:
#        number_of_files(int): number of files for each class of the dataset (ON and OFF),
#                              a number bigger than the number of files per each class 
#                              in the dataset will result in the extraction of all 
#                              available names
#        
#        train_test_ratio(float): ratio between the amount of files used to train
#                                 and test the algorithm, 0.5 will mean that the 
#                                 half of the files wiil be used for training.
#        
#        shuffle_seed(int) : The seed used to shuffle the data, if 0 it will be totally 
#                       random (no seed used)
#       
#        use_all_addr(bool) : if False all off events will be dropped, and the total addresses
#                        number will correspond to the number of channel of the cochlea    
#
#        filenames_train(list of strings) : a list of filenames can be provided to 
#                                           to reproduce the same dataset of a previous
#                                           experiment 
#        filenames_test(list of strings) : a list of filenames can be provided to 
#                                           to reproduce the same dataset of a previous
#                                           experiment 
#        labels_train(list of int) : a list of labels (int) can be provided to 
#                                           to reproduce the same dataset of a previous
#                                           experiment 
#        labels_test(list of int) : a list of labels (int) can be provided to 
#                                           to reproduce the same dataset of a previous
#                                           experiment 
#    Returns:
#        dataset_train(int 2D list): Two equally long lists containing all the 
#                                    timestamps and channel index for training
#        class_train(int list) : The class of each train filename (as a 0 or 1)
#        filenames_test(int 2D list): The filenames plus path for the test files
#        dataset_test(int 2D list): Two equally long lists containing all the 
#                                    timestamps and channel index for testing
#        filenames_train(list of strings) : a list of filenames that can be saved and 
#                                           used to replicate the same experiment
#        filenames_test(list of strings) :  a list of filenames that can be saved and 
#                                           used to replicate the same experiment
# =============================================================================

def on_off_load(number_files_dataset, train_test_ratio, shuffle_seed=0, use_all_addr=False,
                filenames_train=[], filenames_test=[], labels_train=[], labels_test=[]):
    if len(filenames_train)==0 and len(filenames_test)==0 and len(labels_train)==0 and len(labels_test)==0:
        [filenames_train, labels_train, filenames_test, labels_test] = get_filenames_on_off_dataset(number_files_dataset, train_test_ratio, shuffle_seed)
    print ('\n--- READING SPIKES ---')
    start_time = time.time()
    
    dataset_train = []
    dataset_test = []
    
    for train_file in range(len(filenames_train)):
        addresses, timestamps = AERDATA_load(filenames_train[train_file], use_all_addr)
        dataset_train.append([np.array(timestamps), np.array(addresses)])
    for test_file in range(len(filenames_test)):
        addresses, timestamps = AERDATA_load(filenames_test[test_file], use_all_addr)
        dataset_test.append([np.array(timestamps), np.array(addresses)])
      
    
    print("Reading spikes took %s seconds." % (time.time() - start_time))
    
    return dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test

