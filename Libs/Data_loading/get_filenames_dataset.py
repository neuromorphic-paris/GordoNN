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
import glob, os
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

# A simple function that reads file names from datasets, they should be folders 
# subdirectories for every class of the dataset, naming of the subdirectories 
# has to be composed by the name of the class and _aedats (nameclass_aedats)
# =============================================================================
#    Args:
#        dataset_folder(str): string containing the position of the dataset.  
#      
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
#        labels_train(int list): list containing the indexes of the class of the training set
#        labels_test(int list): list containing the indexes of the class of the test set       
#        class_names(string list): the names of the classes indexed as labels_train or test train,
#                                  so that class_names[labels_train[i]] retrieves the name of the class
#                                  for the i-th file in the train set.
# =============================================================================

    
def get_filenames_dataset(dataset_folder, number_of_files = -1, train_test_ratio = 0.75, shuffle_seed = 0):
    print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
    start_time = time.time()
    class_folders = os.listdir(dataset_folder) # Where the dataset is supposed to be placed
    n_classes = len(class_folders)
    # The convention here is nameclass_aedats, so to get the names of the class
    # i need to remove the last 7 letters
    class_names = [class_folders[cl_i][:-7] for cl_i in range(n_classes)]
    
    filenames_train = []
    filenames_test = []
    labels_train = []
    labels_test = []
    
    # Setting the random state for data shuffling
    rng = np.random.RandomState()
    if(shuffle_seed!=0):
        rng.seed(shuffle_seed)

    for i in range(len(class_names)):
        aedats_in_folder = glob.glob(dataset_folder+ '/'+ class_folders[i] + '/*.aedat')
        print ('No. of files of class', class_names[i], ': ', len(aedats_in_folder))

        if number_of_files > 0:
            print ('Func:get_filenames_dataset(): Getting', number_of_files, 'files from the', class_names[i], 'folder')
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
    return filenames_train[ind], filenames_test, labels_train[ind], labels_test, class_names

def get_class_filenames_dataset(dataset_folder, class_names, number_of_files = -1, train_test_ratio = 0.75, shuffle_seed = 0):
    print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
    start_time = time.time()
    n_classes = len(class_names)
    # The convention here is nameclass_aedats, so to get the names of the class
    # i need to remove the last 7 letters
    # class_names = [class_folders[cl_i][:-7] for cl_i in range(n_classes)]
    
    filenames_train = []
    filenames_test = []
    labels_train = []
    labels_test = []
    
    # Setting the random state for data shuffling
    rng = np.random.RandomState()
    if(shuffle_seed!=0):
        rng.seed(shuffle_seed)

    for i in range(n_classes):
        aedats_in_folder = glob.glob(dataset_folder+ '/'+ class_names[i] + '_aedats/*.aedat')
        print ('No. of files of class', class_names[i], ': ', len(aedats_in_folder))

        if number_of_files > 0:
            print ('Func:get_filenames_dataset(): Getting', number_of_files, 'files from the', class_names[i], 'folder')
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
    return filenames_train[ind], filenames_test, labels_train[ind], labels_test, class_names

# A simple function that reads file names from the Free Spoken Digit Dataset 
# https://github.com/Jakobovski/free-spoken-digit-dataset
# Usage
# The test set officially consists of the first 10% of the recordings. Recordings
# numbered 0-4 (inclusive) are in the test and 5-49 are in the training set.
# =============================================================================
#    Args:
#        shuffle_seed : The seed used to shuffle the data, if 0 it will be totally 
#                       random (no seed used)
#    
#    Returns:
#        filenames_train(string list): The filenames plus path for the training files
#        filenames_test(string list): The filenames plus path for the test files
# =============================================================================
    
def get_filenames_digit_dataset(shuffle_seed = 0):
    print ('\n--- GETTING FILENAMES FROM THE DATASET ---')
    start_time = time.time()
    n_digits = 10 # Number of different digits in the dataset
    n_speakers = 6 # Number of different speakers in the dataset
    n_recs = 50 # Number of different recordings per speaker

    folder = 'Data/Spoken_digits' # Where the dataset is supposed to be placed
    
    filenames_train = []
    filenames_test = []


    # Setting the random state for data shuffling
    rng = np.random.RandomState()
    if(shuffle_seed!=0):
        rng.seed(shuffle_seed)
    
    aedats_in_folder = sorted(glob.glob(folder + '/*.aedat'))    
    for set_ind in range(6*10):
        rec_set = aedats_in_folder[set_ind*n_recs:(set_ind+1)*n_recs]
        filenames_test = filenames_test + rec_set[0] + rec_set[1:35:11]
        del rec_set[1:35:11] # removing recording 1 to 4 from training set
        del rec_set[0] # removing recording 0 from training set
        filenames_train = filenames_train + rec_set
        
        #TODO generate labels and then shuffle them
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
