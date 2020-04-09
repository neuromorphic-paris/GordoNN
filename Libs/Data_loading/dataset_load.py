# -*- coding: utf-8 -*-
"""
Created on Mar Jan 15 11:36:42 2019

@author: marcorax, pedro

This file contains functions to automatically loads and shuffles datasets
 
"""
import numpy as np
import time, csv
from Libs.Data_loading.get_filenames_dataset import get_filenames_on_off_dataset
from Libs.Data_loading.AERDATA_load import AERDATA_load


# function to change slashs from win to linux on the label file
def change_slash(label_file):
    rows = []
    with open(label_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            row[0]=row[0].replace('\\', '/')
            #To change naming scheme
            row[0]=row[0].replace('on_aedat', 'on_aedats')
            row[0]=row[0].replace('off_aedat', 'off_aedats')
            rows.append(row)
        csv_file.close()


    # Write data to the csv file and replace the lines in the line_to_override dict.
    with open(label_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)
        csv_file.close()
        
    
# A function used to load the labels for the ON OFF dataset
# =============================================================================
#    Args:
#   
#
#
#    Returns:
#
# =============================================================================
def beg_end_load_on_off(dataset, filenames, label_file):   
    folder_pos="Data/On_Off/"
    begin = np.zeros(len(filenames), dtype=int)
    end = np.zeros(len(filenames), dtype=int)
    count = 0 # number of times a begin and end is written
    with open(label_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            name = folder_pos + row[0]
            files_index = np.where(filenames==name)
            if len(files_index[0])==0:
                continue
            elif len(files_index[0])>1:
                print("Multiple files in filenames with the same name found, something is wrong")
            elif len(files_index[0])==1:
                begin[files_index[0][0]]=int(float(row[1]))
                end[files_index[0][0]]=int(float(row[2]))
                count+=1            
            else:
                print("np.size(files_index) in labels_load_on_off is not a positive integer and i don't know what to do ")
        csv_file.close()
    if count!=len(dataset):
        print("number of files found by labels load different from the filenames list, check all files are present")

    return [begin, end]
        
        
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

def on_off_load(number_files_dataset, label_file,  train_test_ratio, shuffle_seed=0, use_all_addr=False,
                filenames_train=[], filenames_test=[], labels_train=[], labels_test=[]):
    
    if len(filenames_train)==0 and len(filenames_test)==0 and len(labels_train)==0 and len(labels_test)==0:
        [filenames_train, filenames_test, labels_train, labels_test] = get_filenames_on_off_dataset(number_files_dataset, train_test_ratio, shuffle_seed)
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
    
    wordpos_train=beg_end_load_on_off(dataset_train,filenames_train, label_file)
    wordpos_test=beg_end_load_on_off(dataset_test,filenames_test, label_file)
    
    print("Reading spikes took %s seconds." % (time.time() - start_time))
    
    return dataset_train, dataset_test, labels_train, labels_test, filenames_train, filenames_test, wordpos_train, wordpos_test

