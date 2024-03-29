#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:05:23 2022

@author: marcorax93
"""

import numpy as np
import time

from keras.utils.np_utils import to_categorical  


from tensorflow.keras.layers import Input, Dense, BatchNormalization,\
                                    Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras import activations

###METHODS###
def train_mlp(self, layer_dataset, labels):
        """
        Method to process and learn cross features.
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        
        Arguments: 
             layer_dataset: list of individual event based recoriding as
                            generated by the cochlea
                            
             labels: list of labels index of each recording of layer_dataset
                            
        """
        
            
        # Check the runtime mode (multiple batches or single batch)
        n_files = len(layer_dataset)
        
        if self.n_batch_files==None:
            n_batches = 1
            n_runs = 1
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
        
        if n_batches==1:
            n_runs = 1
        else:
            n_runs = self.dataset_runs # how many time a single dataset get cycled.
        
        total_batches  = n_batches*n_runs
        
            
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        input_size = self.features.shape[-1]
        mlp_output_size = np.max(labels)+1
        
        #DEFINE MLP
        mlp = create_mlp(input_size, self.mlp_hidden_size, 
                         mlp_output_size, self.mlp_learning_rate)
                
        for run in range(n_runs):    
            for i_batch_run in range(n_batches):
                
                cross_batch_tv = self.batch_tv_generator(layer_dataset,
                                                         n_batch_files,
                                                         i_batch_run, 
                                                         par_verbose)
                
                rec_ind_1 = i_batch_run*n_batch_files
                rec_ind_2 = (i_batch_run+1)*n_batch_files
                
                labels_subset = labels[rec_ind_1:rec_ind_2]
                
                #Cut to only the last timevectors
                print("MODDATP")
                cross_batch_tv = [cross_batch_tv[i][-500:] for i in range(len(labels_subset))]
                
                ev_labels=[]
                for i in range(len(labels_subset)):
                    ev_labels.append(labels_subset[i]*np.ones(cross_batch_tv[i].shape[0]))
                ev_labels = np.concatenate(ev_labels, axis=0)
                
                ev_labels = to_categorical(ev_labels,num_classes=mlp_output_size)


                # The final results of the local surfaces train dataset computation
                cross_batch_tv = np.concatenate(cross_batch_tv, axis=0)
                cross_batch_tv.reshape(-1, cross_batch_tv.shape[-1])

                
                mlp.fit(cross_batch_tv, ev_labels,
                        batch_size=self.mlp_minibatchsize,
                        epochs=self.mlp_epochs)

                
                if self.verbose is True: 
                    batch_time = time.time()-batch_start_time
                    i_batch = i_batch_run + n_batches*run                    
                    expected_t = batch_time*(total_batches-i_batch-1)
                    total_time += (time.time() - batch_start_time)
                    print("Batch %i out of %i processed, %s seconds left "\
                          %(i_batch+1,total_batches,expected_t))                
                    batch_start_time = time.time()
            

        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))
        
        self.mlp_classifier = mlp

def test_mlp(self, layer_dataset, labels):
        """
        Method to process and learn cross features.
        Since datasets can be memory intensive to process this method works 
        with minibatch clustering, if the number of batch per files 
        
        Arguments: 
             layer_dataset: list of individual event based recoriding as
                            generated by the cochlea
                            
             labels: list of labels index of each recording of layer_dataset
                            
        """
        
            
        # Check the runtime mode (multiple batches or single batch)
        n_files = len(layer_dataset)
        
        if self.n_batch_files==None:
            n_batches = 1
            n_batch_files = n_files
        else:
            n_batch_files = self.n_batch_files
            # number of batches per run   
            n_batches=int(np.ceil(n_files/n_batch_files))  
                     
            
        
        #Set the verbose parameter for the parallel function. #TODO set outside layer
        if self.verbose:
            par_verbose = 0
           # print('\n--- LAYER '+str(layer)+' CROSS TIME VECTORS LEARNING ---')
            batch_start_time = time.time()
            total_time = batch_start_time-batch_start_time
        else:
            par_verbose = 0
            
        mlp_output_size = np.max(labels)+1
        
        #DEFINE MLP
        mlp = self.mlp_classifier
        accuracy = 0
        
        for i_batch_run in range(n_batches):
            
            cross_batch_tv = self.batch_tv_generator(layer_dataset,
                                                     n_batch_files,
                                                     i_batch_run, 
                                                     par_verbose)
            
            rec_ind_1 = i_batch_run*n_batch_files
            rec_ind_2 = (i_batch_run+1)*n_batch_files
            
            labels_subset = labels[rec_ind_1:rec_ind_2]
            
            #Cut to only the last timevectors
            cross_batch_tv = [cross_batch_tv[i][-500:] for i in range(len(labels_subset))]
                
                
            ev_labels=[]
            for i in range(len(labels_subset)):
                ev_labels.append(np.ones(cross_batch_tv[i].shape[0]))
            ev_labels = np.concatenate(ev_labels, axis=0)
            
            ev_labels = to_categorical(ev_labels,num_classes=mlp_output_size)


            # The final results of the local surfaces train dataset computation
            cross_batch_tv = np.concatenate(cross_batch_tv, axis=0)
            cross_batch_tv.reshape(-1, cross_batch_tv.shape[-1])


            
            accuracy+=mlp.predict(cross_batch_tv)
            
            

        accuracy = accuracy/n_batches
        if self.verbose is True:    
            print("learning time vectors took %s seconds." % (total_time))
            print("Accuracy: "+str(accuracy))
                        
        return accuracy
            
# =============================================================================
def create_mlp(input_size, hidden_size, output_size, learning_rate):
    """
    Function used to create a small mlp used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : list of length n containing the sizes for the n
                            layers
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        mlp (keras model) : the freshly baked network
    """
    def relu_advanced(x):
        return activations.relu(x, alpha=0.3)
    
    print("Network Creation")
    inputs = Input(shape=(input_size,), name='encoder_input')
    # x = BatchNormalization()(inputs)
    n_layers = len(hidden_size)
    if n_layers>0:
        for i in range(n_layers):
            if i==0:
                x = Dense(hidden_size[i], activation=relu_advanced)(inputs)
            else:
                x = Dense(hidden_size[i], activation=relu_advanced)(x)
                # x = Dropout(0.7)(x)#0.7
                
        outputs = Dense(output_size, activation='softmax')(x)
    else:
        outputs = Dense(output_size, activation='softmax')(inputs)
    
    
    # adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    # mlp.compile(optimizer=adam,
    #           loss='categorical_crossentropy', metrics=['accuracy'])
    mlp.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])    
    return mlp      