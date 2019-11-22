"""
Created on Mar Nov 12 13:55:14 2019

@author: jpdominguez, marcorax
    
"""



# General Porpouse Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import datetime
import gc
import cv2

# Data loading Libraries
from Libs.Data_loading.dataset_load import on_off_load

# 3D Dimensional HOTS or Solid HOTS
from Libs.Solid_HOTS.Solid_HOTS_Network import Solid_HOTS_Net

# Import NN-related libraries
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from keras import models
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K 

def run_NN():
    batch_size = 32
    num_epochs = 30

    train_datagen   =    ImageDataGenerator(rescale=1./255, validation_split=0.15)
    test_datagen    =    ImageDataGenerator(rescale=1./255)

    # Adapting paths to match images
    filenames_train_processed = filenames_train.copy()
    filenames_train_processed = [path.replace("On_Off", "On_Off_images") for path in filenames_train_processed]
    filenames_train_processed = [path.replace(".wav.aedat", ".png") for path in filenames_train_processed]
    filenames_test_processed = filenames_test.copy()
    filenames_test_processed = [path.replace("On_Off", "On_Off_images") for path in filenames_test_processed]
    filenames_test_processed = [path.replace(".wav.aedat", ".png") for path in filenames_test_processed]


    images_train = []
    images_test = []
    labels_train = classes_train.tolist().copy()
    labels_test = classes_test.tolist().copy()
    cont = 0
    
    for i in range(len(filenames_train_processed)):
        try:
            images_train.append(cv2.resize(cv2.imread(filenames_train_processed[i], cv2.IMREAD_COLOR), (64, 52), interpolation=cv2.INTER_CUBIC))
        except:            
            #labels_train.remove(i)
            del labels_train[i-cont]
            cont+=1
            print(str(cont))

    cont = 0
    for i in range(len(filenames_test_processed)):
        try:
            images_test.append(cv2.resize(cv2.imread(filenames_test_processed[i], cv2.IMREAD_COLOR), (64, 52), interpolation=cv2.INTER_CUBIC))
        except:            
            #labels_test.remove(i)
            del labels_test[i-cont]
            cont+=1
            print(str(cont))


    images_train = np.array(images_train)
    images_test = np.array(images_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)


    train_datagenerator = train_datagen.flow(images_train, labels_train, batch_size=batch_size)
    test_datagenerator = test_datagen.flow(images_test, labels_test, batch_size=batch_size)

    gc.collect()


    # Network model
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(6, 5, input_shape=(52, 64, 3)))
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.MaxPooling2D(pool_size=2))

    CNN.add(layers.Conv2D(14, 3))
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.MaxPooling2D(pool_size=2))

    CNN.add(layers.Conv2D(16, 3))
    CNN.add(layers.Activation('relu'))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(480))
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(480))
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dense(1))
    CNN.add(layers.Activation('sigmoid'))


    optimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    CNN.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = CNN.fit_generator(train_datagenerator, steps_per_epoch = len(images_train) // batch_size, epochs = num_epochs, validation_data=test_datagenerator, validation_steps=len(images_test) // batch_size)

    val_acc = history.history['val_acc']

    return np.amax(val_acc)

# Parameters
    #%% ON OFF Dataset
    # Two class of recordings are used. The first class is composed by files containing
    # a single word each, "ON", the second class is equal but the spelled word is "OFF"
    # =============================================================================
    # number_files_dataset : the number of files to be loaded for each class (On, Off)
    # train_test_ratio: ratio between the amount of files used to train
    #                                 and test the algorithm, 0.5 will mean that the 
    #                                 half of the files wiil be used for training.
    # use_all_addr : if False all off events will be dropped, and the total addresses
    #                number will correspond to the number of channel of the cochlea
    # =============================================================================
number_files_dataset = 200
train_test_ratio = 0.80
use_all_addr = False
number_of_labels = 2
parameter_folder = "Parameters/On_Off/"
label_file = "Data/On_Off/files_timestamps.csv"
legend = ("On","Off") # Legend containing the labes used for plots

NUM_TESTS = 20


NN_results = []


for i in range(NUM_TESTS):

    ### Selecting the dataset
    shuffle_seed = 1 # seed used for dataset shuffling if set to 0 the process will be totally random
    [dataset_train, dataset_test, classes_train, classes_test, filenames_train, filenames_test, wordpos_train, wordpos_test] = on_off_load(number_files_dataset, label_file, train_test_ratio, shuffle_seed, use_all_addr)

    #acc_HOTS = run_HOTS()
    acc_NN = run_NN()
    NN_results.append(acc_NN)
    
    #print("Accuracy HOTS: " + str(acc_HOTS) + ".   Accuracy NN: " + str(acc_NN) + ".")

print(NN_results)


def run_HOTS():

    # To avoid MKL inefficient multythreading
    os.environ['MKL_NUM_THREADS'] = '1'
    
    #%% Network setting and feature exctraction 
    
    # Network settings
    # =============================================================================
    #   features_number (nested lists of int) : the number of feature or centers used by the Solid network,
    #                               the first index identifies the layer, the second one
    #                               is 0 for the centers of the 0D sublayer, and 1 for 
    #                               the 2D centers
    #   context_lengths (list of int): the length of the time context generatef per each layer
    #   input_channels (int) : the total number of channels of the cochlea in the input files 
    #   taus_T(list of float lists) :  a list containing the time coefficient used for 
    #                                  the context creations for each layer (first index)
    #                                  and each channel (second index) 
    #   taus_2D (list of float) : a list containing the time coefficients used for the 
    #                            creation of timesurfaces per each layer
    #   threads (int) : The network can compute timesurfaces in a parallel way,
    #                   this parameter set the number of multiple threads allowed to run
    #   exploring (boolean) : If True, the network will output messages to inform the 
    #                         the users about the current states and will save the 
    #                         basis at each update to build evolution plots (currently not 
    #                         available cos the learning is offline)
    # =============================================================================
    
    
    features_number = [[10,20]]
    context_lengths = [500,200,200]
    input_channels = 32 + 32*use_all_addr
    l1_norm_coeff=[[1e-5,1e-5],[1e-5,1e-5],[1e-5,1e-5]]
    
    #channel_taus = np.array([45, 56, 70, 88, 111, 139, 175, 219, 275, 344, 432, 542, 679, 851, 1067,
    #                         1337, 1677, 2102, 2635, 3302, 4140, 5189, 6504, 8153, 10219, 12809, 16056,
    #                         20126, 25227, 31621, 39636, 49682]) # All the different tau computed for the particular 
    #                                                             # cochlea used for this datasets
    
    channel_taus = np.ones(32)*15000
                                                                 
    second_layer_taus = np.ones(features_number[0][1]) # The taus for this layer are homogeneous across all channels
    #third_layer_taus = np.ones(features_number[1][1]) # The taus for this layer are homogeneous across all channels
    taus_T_coeff = np.array([50,500000]) # Multiplicative coefficients to help to change quickly the taus_T
    
    taus_T = (taus_T_coeff*[channel_taus, second_layer_taus]).tolist()
    taus_2D = [5000,0,0]  
    

    # Create the network
    Net = Solid_HOTS_Net(features_number, context_lengths, input_channels, taus_T, taus_2D, threads=12, exploring=True)    
    learning_rate = [[1e-5,1e-4],[1e-4,5e-4],[1e-4,5e-4]]
    epochs = [[80,100],[10,40],[40,40]]
    

    # Learn the feature
    Net.learn(dataset_train,learning_rate, epochs, l1_norm_coeff)
    """
    tmpcr_c=Net.tmpcr_c
    tmporig_c=Net.tmporig_c
    tmpcr=Net.tmpcr
    tmporig=Net.tmporig
    """
    gc.collect()

    
    #%% Mlp classifier training    
    number_of_labels=len(legend)
    mlp_learning_rate = 8e-4
    Net.mlp_single_word_classification_train(classes_train, wordpos_train, number_of_labels, mlp_learning_rate)
    gc.collect()


    #%% Mlp classifier testing      
    prediction_rate, predicted_labels, predicted_labels_exv = Net.mlp_single_word_classification_test(classes_test, number_of_labels, 0.8, dataset_test)
    print('Prediction rate is '+str(prediction_rate*100)+'%') 
    
    return prediction_rate


