#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dic 1 13:51:42 2018

@author: pedro, marcorax

This file contain functions to extract filenames that will be loaded in AERDATA 
objects.
At this moment it contains only a function for the "on and off" dataset 
but in the case of additional tests with different datasets it will be updated.
 
"""
import numpy as np
import random
import glob
import math
import time


# A simple function that reads file names from a binary dataset (two words:
# "On" and "Off") 
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
#        shuffle_seed : The seed used to shuffle the data, if 0 it will be totally 
#                       random (no seed used)
#    
#    Returns:
#        filenames_train(string list): The filenames plus path for the training files
#        filenames_test(string list): The filenames plus path for the test files
# =============================================================================
    
def get_filenames_on_off_dataset(number_of_files = -1, train_test_ratio = 0.75, shuffle_seed = 0):
    print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
    start_time = time.time()
    used_classes = ['off', 'on']

    
    folders = ['Data/On_Off/off_aedats', 'Data/On_Off/on_aedats'] # Where the dataset is supposed to be placed
    
    filenames_train = []
    filenames_test = []
    labels_train = []
    labels_test = []

    # Setting the random state for data shuffling
    rng = np.random.RandomState()
    if(shuffle_seed!=0):
        rng.seed(shuffle_seed)

    for i in range(len(used_classes)):
        aedats_in_folder = glob.glob(folders[i] + '/*.aedat')
        print ('No. of files of class', used_classes[i], ': ', len(aedats_in_folder))

        if number_of_files > 0:
            print ('Func:get_filenames_dataset(): Getting', number_of_files, 'files from the', used_classes[i], 'folder')
            aedats_in_folder = rng.choice(aedats_in_folder, number_of_files, replace=False)
        elif number_of_files > len(aedats_in_folder):
            print ('Func:get_filenames_dataset(): Error: the number of files selected is bigger than the number of .aedat file in the folder. Getting the whole dataset')
    
        aedats_for_training = int(math.ceil(len(aedats_in_folder)*train_test_ratio))
    
        for ind_train in range(aedats_for_training):
            filenames_train.append(aedats_in_folder[ind_train])
            labels_train.append(i)
        for ind_test in range(aedats_for_training, len(aedats_in_folder)):
            filenames_test.append(aedats_in_folder[ind_test])
            labels_test.append(i)
    
    filenames_train=np.asarray(filenames_train)
    filenames_test=np.asarray(filenames_test)
    labels_train=np.asarray(labels_train)
    labels_test=np.asarray(labels_test)
    ind = np.arange(len(filenames_train))
    rng.shuffle(ind)
    
    print("Getting filenames from the dataset took %s seconds." % (time.time() - start_time))
    return filenames_train[ind], filenames_test, labels_train[ind], labels_test