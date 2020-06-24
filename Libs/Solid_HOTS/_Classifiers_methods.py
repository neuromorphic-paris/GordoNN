"""
@author: marcorax

This file contains the Solid_HOTS_Net class methods that are used
to train and tests classifiers to assess the pattern recognition performed by 
HOTS
 
"""

# Computation libraries
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import keras
#from keras.utils import to_categorical

# Homemade Fresh Libraries like Grandma does (careful, it's still hot!)
from ._General_Func import create_mlp
from ._General_Func import create_lstm
from ._General_Func import create_CNN

import pyNAVIS

# Method for training a mlp classification model 
# =============================================================================      
def mlp_classification_train(self, labels, labels_test, number_of_labels, learning_rate,
                             hidden_size, epochs, batch_size, patience=90000000):
    
    """
    Method to train a simple mlp, to a classification task, to test the feature 
    rapresentation automatically extracted by HOTS
    
    Arguments:
        labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        learning_rate (float) : The learning rate used for the backprop method,
                                Adam
        hidden_size (int) : dimension of the hidden layer used for one layered mlp
        epochs (int) : number of training cycles
        batch_size (int) : mlp batchsize
        patience (int) : number of consecutive higher test_loss cycles before 
                         early stopping (90000000 by default)

    """
    
    # Exctracting last layer activity         
    last_layer_activity = self.last_layer_activity.copy()
    last_layer_activity_test = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    num_of_recordings_test=len(last_layer_activity_test)
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])        
    
    # Generate a label for each event of the train dataset and concatenate last
    # layer activity
    labelled_ev=[]
    for record in range(num_of_recordings):
        record_labels=np.zeros([len(last_layer_activity[record][0]),2])
        record_labels[:,labels_trim[record]] = 1
        labelled_ev.append(record_labels)
    labelled_ev=np.concatenate(labelled_ev)
    last_layer_activity_concatenated = np.concatenate([last_layer_activity[recording][1] for recording in range(num_of_recordings)])
    
    # Generate a label for each event of the train dataset and concatenate last
    # layer activity
    labelled_ev_test=[]
    for record in range(num_of_recordings_test):
        record_labels_test=np.ones([len(last_layer_activity_test[record][0]),2])
        record_labels_test[:,labels_trim_test[record]] = 1
        labelled_ev_test.append(record_labels_test)
    labelled_ev_test=np.concatenate(labelled_ev_test)
    last_layer_activity_test_concatenated = np.concatenate([last_layer_activity_test[recording][1] for recording in range(num_of_recordings_test)])
    
    # Generate the mlp
    last_bottlnck_size = self.features_number[-1][1] # Last layer of Solid HOTS 
                                                     # bottleneck size.
    self.mlp = create_mlp(input_size=last_bottlnck_size, hidden_size=hidden_size,
                          output_size=number_of_labels, learning_rate=learning_rate)
    self.mlp.summary()
    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # fit model
    self.mlp.fit(np.array(last_layer_activity_concatenated), labelled_ev,
      epochs=epochs,
      batch_size=batch_size, shuffle=True, validation_data=(np.array(last_layer_activity_test_concatenated),labelled_ev_test),
      callbacks=[es])
        
    if self.exploring is True:
        print("Training ended, you can now access the trained network with the method .mlp")
    
    
    return 



# Method for testing the mlp classification model
# =============================================================================      
def mlp_classification_test(self, labels, number_of_labels, batch_size, threshold = 0.6):
    """
    Method to test the simple mlp built with .mlp_classification_test(),
    in a classification task, to test the feature 
    rapresentation automatically extracted by HOTS
    
    Arguments:
        labels (list of int) : List of integers (labels) of the testing dataset
        number_of_labels (int) : The total number of different labels that,
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        batch_size (int) : mlp batchsize
        threshold (float between 0.5 and 1) : A threshold applied on the
                                            absolute sigmoid output of the,
                                            classifier, anyithing within that 
                                            range will be considered uncertainty,
                                            coded as a -1 label.

    """
   
    # Exctracting last layer activity (test)  
    last_layer_activity = self.last_layer_activity_test.copy()    
    num_of_recordings = len(last_layer_activity)
    
    
    predicted_labels=[]
    net_activity=[]
    
    labels_trim=labels.copy()
    
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim=np.delete(labels_trim, self.abs_rem_ind_test[i])
    
    # concatenate last layer activity and generate predicted label per event
    last_layer_activity_concatenated = np.concatenate([last_layer_activity[recording][1] for recording in range(num_of_recordings)])
    predicted_labels_ev=self.mlp.predict(np.array(last_layer_activity_concatenated),batch_size=batch_size)
    
    events_counter = 0 # counter used to cycle through all concatenated events
    for recording in range(len(last_layer_activity)):
        # Exctracting events per recording
        predicted_labels_recording = predicted_labels_ev[events_counter:\
                                                         events_counter+\
                                                             len(last_layer_activity[recording][0])]
        net_activity.append(predicted_labels_recording)
        predicted_labels_recording_th=(np.asarray(predicted_labels_recording)>=threshold)*np.asarray(predicted_labels_recording)
        predicted_labels_recording_sum = np.sum(predicted_labels_recording_th, 0)    
        if max(predicted_labels_recording_sum) == 0:
            predicted_labels.append(-1)
        else:
            predicted_labels.append(np.argmax(predicted_labels_recording_sum))
        events_counter += len(last_layer_activity[recording][0])
   
    prediction_rate = 0
    prediction_rate_wht_uncert = 0 # Prediction rate without uncertain responses
                                   # in which the classifiers didn't produced 
                                   # any values higher than the threshold
    for i,true_label in enumerate(labels_trim):
        prediction_rate += (predicted_labels[i] == true_label)/len(labels_trim)
        prediction_rate_wht_uncert += (predicted_labels[i] == true_label)/(len(labels_trim)-sum(np.asarray(predicted_labels)==-1))
        
    # Save mlp classifier response for debugging:
    self.mlp_class_response = [predicted_labels, net_activity]
    
    print('Prediction rate is: '+str(prediction_rate*100)+'%') 
    print('Prediction rate without uncertain responses (in which the classifiers did not produced'+\
          ' any values higher than the threshold) is: ' +str(prediction_rate_wht_uncert*100)+'%') 
    
    return prediction_rate, prediction_rate_wht_uncert





# Method for training a histogram mlp classification model 
# =============================================================================      
def hist_mlp_classification_train(self, labels, labels_test, number_of_labels, learning_rate,
                             hidden_size, epochs, batch_size, patience=90000000):
    
    """
    Method to train a simple mlp to classify histograms of Net responses of the
    recording, to test the feature rapresentation automatically extracted by HOTS
    The histogram classifier is based on the original HOTS paper
    
    Arguments:
        labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        learning_rate (float) : The learning rate used for the backprop method,
                                Adam
        hidden_size (int) : dimension of the hidden layer used for one layered mlp
        epochs (int) : number of training cycles
        batch_size (int) : mlp batchsize
        patience (int) : number of consecutive higher test_loss cycles before 
                         early stopping (90000000 by default)

    """
    
          
    # Exctracting last layer activity         
    last_layer_activity = self.last_layer_activity.copy()
    last_layer_activity_test = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    num_of_recordings_test=len(last_layer_activity_test)
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])        
        
    # The histograms for each class, also known as "signatures"     
    cl_hist = np.zeros([num_of_recordings,len(last_layer_activity[0][1][0])])
    cl_hist_test = np.zeros([num_of_recordings_test,len(last_layer_activity[0][1][0])])
    
    # The array of lables in the same structure required to train the mlp
    mlp_labels=np.zeros([len(labels_trim),number_of_labels])
    mlp_labels_test=np.zeros([len(labels_trim_test),number_of_labels])
    
    # Computing the signatures
    for i,cl in enumerate(labels_trim): 
        cl_hist[i,:] = np.mean(last_layer_activity[i][1],0)
        mlp_labels[i,cl] = 1
    
    for i,cl in enumerate(labels_trim_test): 
        cl_hist_test[i,:] = np.mean(last_layer_activity_test[i][1],0)
        mlp_labels_test[i,cl] = 1
    
    
    # Generate the mlp
    last_bottlnck_size = self.features_number[-1][1] # Last layer of Solid HOTS 
                                                     # bottleneck size.
    self.hist_mlp = create_mlp(input_size=last_bottlnck_size, hidden_size=hidden_size,
                          output_size=number_of_labels, learning_rate=learning_rate)
    self.hist_mlp.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
   
    # fit model
    self.hist_mlp.fit(np.array(cl_hist), mlp_labels,
      epochs=epochs,
      batch_size=batch_size, shuffle=True,
      validation_data=(np.array(cl_hist_test),mlp_labels_test),
      callbacks=[es])
        
    if self.exploring is True:
        print("Training ended, you can now access the trained network with the method .hist_mlp")
    
    return 


# Method for testing the histogram mlp classification model
# =============================================================================      
def hist_mlp_classification_test(self, labels, number_of_labels, batch_size, threshold):
    """
    Method to test the simple histogram mlp built with 
    .hist_mlp_classification_test(), in a classification task, to test the 
    feature rapresentation automatically extracted by HOTS
    
    Arguments:
        labels (list of int) : List of integers (labels) of the dataset
        number_of_labels (int) : The total number of different labels that,
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        batch_size (int) : mlp batchsize
        threshold (float between 0.5 and 1) : A threshold applied on the
                                            absolute sigmoid output of the,
                                            classifier, anyithing within that 
                                            range will be considered uncertainty,
                                            coded as a -1 label.

    """
 
    # Exctracting last layer activity (test)        
    last_layer_activity = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    
    labels_trim=labels.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind_test[i])
 
        
    # The histograms for each class, also known as "signatures"     
    cl_hist_test = np.zeros([num_of_recordings,len(last_layer_activity[0][1][0])])
    
    # The array of lables in the same structure required to train the mlp
    mlp_labels_test=np.zeros([len(labels_trim),number_of_labels])
    
    # Computing the signatures   
    for i,cl in enumerate(labels_trim): 
        cl_hist_test[i,:] = np.mean(last_layer_activity[i][1],0)
        mlp_labels_test[i,cl] = 1
    
    # Generating the predicted values
    mlp_predicted_labels=self.hist_mlp.predict(np.array(cl_hist_test),batch_size=batch_size)
    mlp_predicted_labels_th = (mlp_predicted_labels>threshold)*mlp_predicted_labels
    
    predicted_labels=[]
    for recording in range(len(labels_trim)):
        if max(mlp_predicted_labels_th[recording]) == 0:
            predicted_labels.append(-1)
        else:
            predicted_labels.append(np.argmax(mlp_predicted_labels_th[recording]))

    prediction_rate = 0
    prediction_rate_wht_uncert = 0 # Prediction rate without uncertain responses
                                   # in which the classifiers didn't produced 
                                   # any values higher than the threshold
                                   
    for i,true_label in enumerate(labels_trim):
        prediction_rate += (predicted_labels[i] == true_label)/len(labels_trim)
        prediction_rate_wht_uncert += (predicted_labels[i] == true_label)/(len(labels_trim)-sum(np.asarray(predicted_labels)==-1))        
            
    # Save mlp classifier response for debugging:
    self.hist_mlp_class_response = [predicted_labels, cl_hist_test]
    
    print('Prediction rate is: '+str(prediction_rate*100)+'%') 
    print('Prediction rate without uncertain responses (in which the classifiers did not produced'+\
          ' any values higher than the threshold) is: ' +str(prediction_rate_wht_uncert*100)+'%') 
    
    return 



# Method for training a lstm classification model 
# =============================================================================      
def lstm_classification_train(self, dataset_train, labels, dataset_test, labels_test, number_of_labels, bin_width, 
                             sliding_amount, learning_rate, units, epochs, 
                             batch_size, patience=90000000):
    
    """
    Method to train a simple lstm, to a classification task, to test the feature 
    rapresentation automatically extracted by HOTS
    
    Arguments:
        labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        bin_width (int) : number of samples that are considered for a single
                          input bin to the LSTM
        sliding_amount (int) : number of samples that will be skipped from one
                               bin to the next one. Used for overlapping between bins. 
        learning_rate (float) : The learning rate used for the backprop method,
                                Adam
        units (int) : dimension of the LSTM layer
        epochs (int) : number of training cycles
        batch_size (int) : mlp batchsize
        patience (int) : number of consecutive higher test_loss cycles before 
                         early stopping (90000000 by default)

    """
    
    ### Sonogram ###
    
    bin_size = 1000 #microsec
    num_channels = 32
    delta_t = 0
    settings = pyNAVIS.MainSettings(num_channels=num_channels, mono_stereo=0, bin_size=bin_size, on_off_both=0)
    
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])   
    
    
    sonograms_train = []
    for i in range(len(dataset_train)):
        spikes_file = pyNAVIS.SpikesFile(dataset_train[i][1], dataset_train[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        sonogram = sonogram/np.max(sonogram) #Normalize the sonogram between 0 and 1.
        sonograms_train.append(np.transpose(sonogram))
    
    sonograms_test = []
    for i in range(len(dataset_test)):
        spikes_file = pyNAVIS.SpikesFile(dataset_test[i][1], dataset_test[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        sonogram = sonogram/np.max(sonogram) #Normalize the sonogram between 0 and 1.
        sonograms_test.append(np.transpose(sonogram))
    
    
    length_array_train = [sonogram.shape[0] for sonogram in sonograms_train]
    length_array_test = [sonogram.shape[0] for sonogram in sonograms_test]
    maxlength = np.max(length_array_train+length_array_test) # information to be used to pad data
    
    
    data_sonograms_train = []
    data_sonograms_test = []
    

    for i in range(len(sonograms_train)):
        
        if delta_t:
            #calculate delta t
            delta_t_sonogram = range(0, sonograms_train[i].shape[0]*bin_size, bin_size)
            
            #We expand the number of rows in last_layer_activity[i][1] 
            #from n to n+1, in order to include delta_t
            lstm_input = np.zeros((maxlength, num_channels+1))
            lstm_input[:length_array_train[i],:-1] = sonograms_train[i]
            lstm_input[:length_array_train[i],-1] = delta_t_sonogram
        else:
            lstm_input = np.zeros((maxlength, num_channels))
            lstm_input[:length_array_train[i],:] = sonograms_train[i]
        data_sonograms_train.append(lstm_input)
    
    data_sonograms_train=np.array(data_sonograms_train) 
    
    
    for i in range(len(sonograms_test)):
        
        if delta_t:
            #calculate delta t
            delta_t_sonogram = range(0, sonograms_test[i].shape[0]*bin_size, bin_size)
            
            #We expand the number of rows in last_layer_activity[i][1] 
            #from n to n+1, in order to include delta_t
            lstm_input = np.zeros((maxlength, num_channels+1))
            lstm_input[:length_array_test[i],:-1] = sonograms_test[i]
            lstm_input[:length_array_test[i],-1] = delta_t_sonogram
        else:
            lstm_input = np.zeros((maxlength, num_channels))
            lstm_input[:length_array_test[i],:] = sonograms_test[i]
        data_sonograms_test.append(lstm_input)
    
    data_sonograms_test=np.array(data_sonograms_test)     
    
    
    
    
    # Create the bins    
    sonogram_activity_binned_train = []
    sonogram_labels_bins_train = []
    for i in range(len(data_sonograms_train)):
        recording_train = []
        if data_sonograms_train[i].shape[0] >= bin_width: #if bin_width >= data_sonograms_train length for that file
            recording_train.append(np.reshape(data_sonograms_train[i][0:bin_width, :], bin_width*num_channels))
            sonogram_labels_bins_train.append(labels_trim[i])
            for j in range(bin_width - 1 + sliding_amount, data_sonograms_train[i].shape[0], sliding_amount):
                recording_train.append(np.reshape(data_sonograms_train[i][j-bin_width:j, :], bin_width*num_channels))
                sonogram_labels_bins_train.append(labels_trim[i])
        sonogram_activity_binned_train.append(np.array(recording_train))
    
    sonogram_labels_bins_train = np.array(sonogram_labels_bins_train)
    
    
    sonogram_activity_binned_test = []
    sonogram_labels_bins_test = []
    for i in range(len(data_sonograms_test)):
        recording_test = []
        if data_sonograms_test[i].shape[0] >= bin_width: #if bin_width >= data_sonograms_test length for that file
            recording_test.append(np.reshape(data_sonograms_test[i][0:bin_width, :], bin_width*num_channels))
            sonogram_labels_bins_test.append(labels_trim_test[i])
            for j in range(bin_width - 1 + sliding_amount, data_sonograms_test[i].shape[0], sliding_amount):
                recording_test.append(np.reshape(data_sonograms_test[i][j-bin_width:j, :], bin_width*num_channels))
                sonogram_labels_bins_test.append(labels_trim_test[i])
        sonogram_activity_binned_test.append(np.array(recording_test))
    
    sonogram_labels_bins_test = np.array(sonogram_labels_bins_test)
    
    
    #self.tmp = [activity_binned_train,activity_binned_test]              
    timesteps = maxlength//sliding_amount
    features = bin_width*(num_channels)
    self.sonogram_lstm = create_lstm(timesteps=timesteps, features=features,
                            hidden_size=units, learning_rate=learning_rate)
    self.sonogram_lstm.summary()
    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # fit model
    self.sonogram_lstm.fit(np.array(sonogram_activity_binned_train), np.array(labels_trim),
      epochs=epochs,
      batch_size=batch_size, validation_data=(np.array(sonogram_activity_binned_test), np.array(labels_trim_test)),
      callbacks=[es], verbose=1)
    
    
    
    
    
    
    ### GORDONN ###
    
    
    # Exctracting last layer activity         
    last_layer_activity = self.last_layer_activity.copy()
    last_layer_activity_test = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    num_of_recordings_test=len(last_layer_activity_test)
    
    data_train = []
    data_test = []
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])      

    length_array_train = [len(last_layer_activity[i][0]) for i in range(num_of_recordings)]
    length_array_test = [len(last_layer_activity_test[i][0]) for i in range(num_of_recordings_test)]
   
    maxlength = np.max(length_array_train+length_array_test) # information to be used to pad data
    
    last_bottlnck_size = self.features_number[-1][1] # Last layer of Solid HOTS 
                                                     # bottleneck size.    
    
    for recording in range(num_of_recordings):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity[recording][0][event] - last_layer_activity[recording][0][event-1]
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from n to n+1, in order to include delta_t
        lstm_input = np.zeros((maxlength, last_bottlnck_size+1))
        lstm_input[:length_array_train[recording],:-1] = last_layer_activity[recording][1]
        lstm_input[:length_array_train[recording],-1] = delta_t_recording
        data_train.append(lstm_input)
    
    data_train=np.array(data_train)      
                                                    
    for recording in range(num_of_recordings_test):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity_test[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity_test[recording][0][event] - last_layer_activity_test[recording][0][event-1]
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from n to n+1, in order to include delta_t
        lstm_input = np.zeros((maxlength, last_bottlnck_size+1))
        lstm_input[:length_array_test[recording],:-1] = last_layer_activity_test[recording][1]
        lstm_input[:length_array_test[recording],-1] = delta_t_recording
        data_test.append(lstm_input)
          
    data_test=np.array(data_test)                                      
                      
    # Create the bins    
    activity_binned_train = []
    labels_bins_train = []
    for i in range(len(data_train)):
        recording_train = []
        if data_train[i].shape[0] >= bin_width: #if bin_width >= data_train length for that file
            recording_train.append(np.reshape(data_train[i][0:bin_width, :], bin_width*(last_bottlnck_size+1)))
            labels_bins_train.append(labels_trim[i])
            for j in range(bin_width - 1 + sliding_amount, data_train[i].shape[0], sliding_amount):
                recording_train.append(np.reshape(data_train[i][j-bin_width:j, :], bin_width*(last_bottlnck_size+1)))
                labels_bins_train.append(labels_trim[i])
        activity_binned_train.append(np.array(recording_train))
        # else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
        #     z = np.zeros([bin_width, data_train[i].shape[1]])
        #     z[:data_train[i].shape[0],:data_train[i].shape[1]] = data_train[i]
        #     activity_binned_train.append(z)
        #     labels_bins_train.append(labels_trim[i])
    
    labels_bins_train = np.array(labels_bins_train)


    activity_binned_test = []
    labels_bins_test = []
    for i in range(len(data_test)):
        recording_test=[]
        if data_test[i].shape[0] >= bin_width: #if bin_width >= data_test length for that file
            recording_test.append(np.reshape(data_test[i][0:bin_width, :], bin_width*(last_bottlnck_size+1)))
            labels_bins_test.append(labels_trim_test[i])
            for j in range(bin_width - 1 + sliding_amount, data_test[i].shape[0], sliding_amount):
                recording_test.append(np.reshape(data_test[i][j-bin_width:j, :], bin_width*(last_bottlnck_size+1)))
                labels_bins_test.append(labels_trim_test[i])
        activity_binned_test.append(np.array(recording_test))
        # else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
        #     z = np.zeros([bin_width, data_test[i].shape[1]])
        #     z[:data_test[i].shape[0],:data_test[i].shape[1]] = data_test[i]
        #     activity_binned_test.append(z)
        #     labels_bins_test.append(labels_trim_test[i])
    
    labels_bins_test = np.array(labels_bins_test)                                                     
                  
                    
    self.tmp = [activity_binned_train,activity_binned_test]              
    timesteps = maxlength//sliding_amount
    features = bin_width*(last_bottlnck_size+1)
    self.lstm = create_lstm(timesteps=timesteps, features=features,
                            hidden_size=units, learning_rate=learning_rate)
    self.lstm.summary()
    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # fit model
    self.lstm.fit(np.array(activity_binned_train), np.array(labels_trim),
      epochs=epochs,
      batch_size=batch_size, validation_data=(np.array(activity_binned_test), np.array(labels_trim_test)),
      callbacks=[es], verbose=1)
        
    return 

# Method for training a lstm classification model 
# =============================================================================      
def lstm_classification_test(self, dataset_train, labels, dataset_test, labels_test, number_of_labels, bin_width, 
                             sliding_amount, batch_size, threshold ):
    
    """
    Method to test a previously-trained LSTM, to a classification task, to test the feature 
    rapresentation automatically extracted by HOTS
    
    Arguments:
        labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        bin_width (int) : number of samples that are considered for a single
                          input bin to the LSTM
        sliding_amount (int) : number of samples that will be skipped from one
                               bin to the next one. Used for overlapping between bins. 
        learning_rate (float) : The learning rate used for the backprop method,
                                Adam
        units (int) : dimension of the LSTM layer
        epochs (int) : number of training cycles
        batch_size (int) : mlp batchsize
        patience (int) : number of consecutive higher test_loss cycles before 
                         early stopping (90000000 by default)

    """
    
    bin_size = 1000 #microsec
    num_channels = 32
    delta_t = 0
    settings = pyNAVIS.MainSettings(num_channels=num_channels, mono_stereo=0, bin_size=bin_size, on_off_both=0)
    
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])   
    
    
    sonograms_train = []
    for i in range(len(dataset_train)):
        spikes_file = pyNAVIS.SpikesFile(dataset_train[i][1], dataset_train[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        sonogram = sonogram/np.max(sonogram) #Normalize the sonogram between 0 and 1.
        sonograms_train.append(np.transpose(sonogram))
    
    sonograms_test = []
    for i in range(len(dataset_test)):
        spikes_file = pyNAVIS.SpikesFile(dataset_test[i][1], dataset_test[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        sonogram = sonogram/np.max(sonogram) #Normalize the sonogram between 0 and 1.
        sonograms_test.append(np.transpose(sonogram))
    
    
    length_array_train = [sonogram.shape[0] for sonogram in sonograms_train]
    length_array_test = [sonogram.shape[0] for sonogram in sonograms_test]
    maxlength = np.max(length_array_train+length_array_test) # information to be used to pad data
    
    
    data_sonograms_train = []
    data_sonograms_test = []
    

    for i in range(len(sonograms_train)):
        
        if delta_t:
            #calculate delta t
            delta_t_sonogram = range(0, sonograms_train[i].shape[0]*bin_size, bin_size)
            
            #We expand the number of rows in last_layer_activity[i][1] 
            #from n to n+1, in order to include delta_t
            lstm_input = np.zeros((maxlength, num_channels+1))
            lstm_input[:length_array_train[i],:-1] = sonograms_train[i]
            lstm_input[:length_array_train[i],-1] = delta_t_sonogram
        else:
            lstm_input = np.zeros((maxlength, num_channels))
            lstm_input[:length_array_train[i],:] = sonograms_train[i]
        data_sonograms_train.append(lstm_input)
    
    data_sonograms_train=np.array(data_sonograms_train) 
    
    
    for i in range(len(sonograms_test)):
        
        if delta_t:
            #calculate delta t
            delta_t_sonogram = range(0, sonograms_test[i].shape[0]*bin_size, bin_size)
            
            #We expand the number of rows in last_layer_activity[i][1] 
            #from n to n+1, in order to include delta_t
            lstm_input = np.zeros((maxlength, num_channels+1))
            lstm_input[:length_array_test[i],:-1] = sonograms_test[i]
            lstm_input[:length_array_test[i],-1] = delta_t_sonogram
        else:
            lstm_input = np.zeros((maxlength, num_channels))
            lstm_input[:length_array_test[i],:] = sonograms_test[i]
        data_sonograms_test.append(lstm_input)
    
    data_sonograms_test=np.array(data_sonograms_test)     
    
    
    
    
    # Create the bins    
    sonogram_activity_binned_train = []
    sonogram_labels_bins_train = []
    for i in range(len(data_sonograms_train)):
        recording_train = []
        if data_sonograms_train[i].shape[0] >= bin_width: #if bin_width >= data_sonograms_train length for that file
            recording_train.append(np.reshape(data_sonograms_train[i][0:bin_width, :], bin_width*num_channels))
            sonogram_labels_bins_train.append(labels_trim[i])
            for j in range(bin_width - 1 + sliding_amount, data_sonograms_train[i].shape[0], sliding_amount):
                recording_train.append(np.reshape(data_sonograms_train[i][j-bin_width:j, :], bin_width*num_channels))
                sonogram_labels_bins_train.append(labels_trim[i])
        sonogram_activity_binned_train.append(np.array(recording_train))
    
    sonogram_labels_bins_train = np.array(sonogram_labels_bins_train)
    
    
    sonogram_activity_binned_test = []
    sonogram_labels_bins_test = []
    for i in range(len(data_sonograms_test)):
        recording_test = []
        if data_sonograms_test[i].shape[0] >= bin_width: #if bin_width >= data_sonograms_test length for that file
            recording_test.append(np.reshape(data_sonograms_test[i][0:bin_width, :], bin_width*num_channels))
            sonogram_labels_bins_test.append(labels_trim_test[i])
            for j in range(bin_width - 1 + sliding_amount, data_sonograms_test[i].shape[0], sliding_amount):
                recording_test.append(np.reshape(data_sonograms_test[i][j-bin_width:j, :], bin_width*num_channels))
                sonogram_labels_bins_test.append(labels_trim_test[i])
        sonogram_activity_binned_test.append(np.array(recording_test))
    
    sonogram_labels_bins_test = np.array(sonogram_labels_bins_test)
    
    
    # fit model
    predicted_labels_sonogram_ev_train = self.sonogram_lstm.predict(np.array(sonogram_activity_binned_train))
    predicted_labels_sonogram_ev_test = self.sonogram_lstm.predict(np.array(sonogram_activity_binned_test))
    
    
    prediction_rate_train_sonogram=sum(labels_trim==np.around(np.reshape(predicted_labels_sonogram_ev_train,len(predicted_labels_sonogram_ev_train))))/len(labels_trim)
    prediction_rate_test_sonogram=sum(labels_trim_test==np.around(np.reshape(predicted_labels_sonogram_ev_test,len(predicted_labels_sonogram_ev_test))))/len(labels_trim_test)
    
    print('Train Prediction rate is: '+str(prediction_rate_train_sonogram*100)+'%') 
    print('Test Prediction rate is: '+str(prediction_rate_test_sonogram*100)+'%')
    
    
    
    
    
    ### GORDONN ###
    
    # Exctracting last layer activity         
    last_layer_activity = self.last_layer_activity.copy()
    last_layer_activity_test = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    num_of_recordings_test=len(last_layer_activity_test)
    
    data_train = []
    data_test = []
    
    labels_trim=labels.copy()
    labels_trim_test=labels_test.copy()
 
    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])      

    length_array_train = [len(last_layer_activity[i][0]) for i in range(num_of_recordings)]
    length_array_test = [len(last_layer_activity_test[i][0]) for i in range(num_of_recordings_test)]
   
    maxlength = np.max(length_array_train+length_array_test) # information to be used to pad data
    
    last_bottlnck_size = self.features_number[-1][1] # Last layer of Solid HOTS 
                                                     # bottleneck size.    
    
    for recording in range(num_of_recordings):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity[recording][0][event] - last_layer_activity[recording][0][event-1]
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from n to n+1, in order to include delta_t
        lstm_input = np.zeros((maxlength, last_bottlnck_size+1))
        lstm_input[:length_array_train[recording],:-1] = last_layer_activity[recording][1]
        lstm_input[:length_array_train[recording],-1] = delta_t_recording
        data_train.append(lstm_input)
    
    data_train=np.array(data_train)      
                                                    
    for recording in range(num_of_recordings_test):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity_test[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity_test[recording][0][event] - last_layer_activity_test[recording][0][event-1]
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from n to n+1, in order to include delta_t
        lstm_input = np.zeros((maxlength, last_bottlnck_size+1))
        lstm_input[:length_array_test[recording],:-1] = last_layer_activity_test[recording][1]
        lstm_input[:length_array_test[recording],-1] = delta_t_recording
        data_test.append(lstm_input)
          
    data_test=np.array(data_test)                                      
                      
    # Create the bins    
    activity_binned_train = []
    labels_bins_train = []
    for i in range(len(data_train)):
        recording_train = []
        if data_train[i].shape[0] >= bin_width: #if bin_width >= data_train length for that file
            recording_train.append(np.reshape(data_train[i][0:bin_width, :], bin_width*(last_bottlnck_size+1)))
            labels_bins_train.append(labels_trim[i])
            for j in range(bin_width - 1 + sliding_amount, data_train[i].shape[0], sliding_amount):
                recording_train.append(np.reshape(data_train[i][j-bin_width:j, :], bin_width*(last_bottlnck_size+1)))
                labels_bins_train.append(labels_trim[i])
        activity_binned_train.append(np.array(recording_train))
        # else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
        #     z = np.zeros([bin_width, data_train[i].shape[1]])
        #     z[:data_train[i].shape[0],:data_train[i].shape[1]] = data_train[i]
        #     activity_binned_train.append(z)
        #     labels_bins_train.append(labels_trim[i])
    
    labels_bins_train = np.array(labels_bins_train)


    activity_binned_test = []
    labels_bins_test = []
    for i in range(len(data_test)):
        recording_test=[]
        if data_test[i].shape[0] >= bin_width: #if bin_width >= data_test length for that file
            recording_test.append(np.reshape(data_test[i][0:bin_width, :], bin_width*(last_bottlnck_size+1)))
            labels_bins_test.append(labels_trim_test[i])
            for j in range(bin_width - 1 + sliding_amount, data_test[i].shape[0], sliding_amount):
                recording_test.append(np.reshape(data_test[i][j-bin_width:j, :], bin_width*(last_bottlnck_size+1)))
                labels_bins_test.append(labels_trim_test[i])
        activity_binned_test.append(np.array(recording_test))
        # else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
        #     z = np.zeros([bin_width, data_test[i].shape[1]])
        #     z[:data_test[i].shape[0],:data_test[i].shape[1]] = data_test[i]
        #     activity_binned_test.append(z)
        #     labels_bins_test.append(labels_trim_test[i])
    
    labels_bins_test = np.array(labels_bins_test)                                                        
                  
    # fit model
    predicted_labels_ev_train = self.lstm.predict(np.array(activity_binned_train))
    predicted_labels_ev_test = self.lstm.predict(np.array(activity_binned_test))
    # # Compute accuracy
    # net_activity =[]
    # predicted_labels=[]
    # event_counter = 0
    
    # for recording in range(len(n_events)):
    #     # Exctracting events per recording
    #     predicted_labels_recording = predicted_labels_ev[event_counter:event_counter+n_events[recording]]
    #     event_counter+=n_events[recording]
    #     net_activity.append(predicted_labels_recording)
    #     predicted_labels_recording_pos=(np.asarray(predicted_labels_recording)>=threshold)*np.asarray(predicted_labels_recording)
    #     predicted_labels_recording_neg=(np.asarray(1-predicted_labels_recording)>=threshold)*np.asarray(1-predicted_labels_recording)
    #     predicted_labels_recording_pos_sum = np.sum(predicted_labels_recording_pos)    
    #     predicted_labels_recording_neg_sum = np.sum(predicted_labels_recording_neg)  
    #     if (predicted_labels_recording_neg_sum+predicted_labels_recording_pos_sum) == 0:
    #         predicted_labels.append(-1)
    #     elif(predicted_labels_recording_pos_sum>predicted_labels_recording_neg_sum):
    #         predicted_labels.append(1)
    #     else:
    #          predicted_labels.append(0)
   
    # prediction_rate = 0
    # prediction_rate_wht_uncert = 0 # Prediction rate without uncertain responses
    #                                # in which the classifiers didn't produced 
    #                                # any values higher than the threshold
    # for i,true_label in enumerate(labels_trim_test):
    #     prediction_rate += (predicted_labels[i] == true_label)/len(labels_trim_test)
    #     prediction_rate_wht_uncert += (predicted_labels[i] == true_label)/(len(labels_trim_test)-sum(np.asarray(predicted_labels)==-1))
        
    # Save mlp classifier response for debugging:
    self.lstm_class_response = [predicted_labels_ev_train, predicted_labels_ev_test]
    prediction_rate_train=sum(labels_trim==np.around(np.reshape(predicted_labels_ev_train,len(predicted_labels_ev_train))))/len(labels_trim)
    prediction_rate_test=sum(labels_trim_test==np.around(np.reshape(predicted_labels_ev_test,len(predicted_labels_ev_test))))/len(labels_trim_test)
    
    print('Train Prediction rate is: '+str(prediction_rate_train*100)+'%') 
    print('Test Prediction rate is: '+str(prediction_rate_test*100)+'%') 
    # print('Prediction rate is: '+str(prediction_rate*100)+'%') 
    # print('Prediction rate without uncertain responses (in which the classifiers did not produced'+\
    #       ' any values higher than the threshold) is: ' +str(prediction_rate_wht_uncert*100)+'%') 
        
        
        
     
    return



# Method for training a cnn classification model 
# =============================================================================      
def cnn_classification_train(self, dataset_train, labels_train, dataset_test,
                             labels_test, number_of_labels, learning_rate,
                             epochs, batch_size, bin_size, patience=90000000):
    
    """
    Method to train a simple cnn, to a classification task, to compare the 
    feature extraction from the CNN with the feature representation 
    automatically extracted by HOTS
    
    Arguments:
        dataset_train (list): list of lists with timestamps and addresses of all training files
        labels_train (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        learning_rate (float) : The learning rate used for the backprop method,
                                Adam
        epochs (int) : number of training cycles
        batch_size (int) : mlp batchsize
        bin_size (int) : size (microseconds) of the window to integrate the spikes when 
                        calculating the sonogram
        patience (int) : number of consecutive higher test_loss cycles before 
                         early stopping (90000000 by default)

    """
    
    ### Sonogram ###
    train_images = []
    train_labels = []
    
    
    settings = pyNAVIS.MainSettings(num_channels=32, mono_stereo=0, bin_size=bin_size,on_off_both=0)
    
    for i in range(len(dataset_train)):
        spikes_file = pyNAVIS.SpikesFile(dataset_train[i][1], dataset_train[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        
        n_bins = 1000000//bin_size #number of bins that the sonogram should have (audios are 1 s long at maximum)
        
        if sonogram.shape[1] < n_bins:
            m_zeros = np.zeros((32, n_bins - sonogram.shape[1]))
            sonogram = np.concatenate((sonogram, m_zeros), axis=1)
        elif sonogram.shape[1] > n_bins:
            sonogram = sonogram[...,:-(sonogram.shape[1]-n_bins)]
        
        train_images.append(sonogram/np.max(sonogram)) #Normalize the sonogram between 0 and 1.
        train_labels.append(labels_train[i])
        
        
        
    
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_images, train_labels, test_size=0.2, random_state=1)
    
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    
    X_val = np.asarray(X_val)
    X_val = X_val.reshape((-1, settings.num_channels, n_bins, 1))
    X_train = np.asarray(X_train)
    X_train = X_train.reshape((-1, settings.num_channels, n_bins, 1))
        
    self.sonogram_cnn = create_CNN(learning_rate=learning_rate, width=n_bins, height=settings.num_channels)
    self.sonogram_cnn.summary()
    
    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # fit model
    self.sonogram_cnn.fit(X_train, y_train, 
      steps_per_epoch= X_train.shape[0]/batch_size,
      epochs=epochs, batch_size=batch_size, shuffle=True, 
      validation_data=(X_val, y_val),
      callbacks=[es])

    if self.exploring is True:
        print("Sonogram training ended, you can now access the trained network with the method .sonogram_cnn")
        
        ##### GORDONN CLASSIFIER ######
    
    # Exctracting last layer activity         
    last_layer_activity = self.last_layer_activity.copy()
    last_layer_activity_test = self.last_layer_activity_test.copy()
    num_of_recordings=len(last_layer_activity)
    num_of_recordings_test=len(last_layer_activity_test)

    for i in range(len(last_layer_activity)):
        #calcualte delta t
        delta_t_i = np.empty(last_layer_activity[i][0].shape[0], dtype=int)
        delta_t_i[0] = int(0)
        for j in range(1, len(delta_t_i)):
            delta_t_i[j] = last_layer_activity[i][0][j] - last_layer_activity[i][0][j-1]
    data_train = []
    data_test = []

    labels_trim=labels_train.copy()
    labels_trim_test=labels_test.copy()

    # remove the labels of discarded files from the method .learn
    for i in range(len(self.abs_rem_ind)-1,-1,-1):
        labels_trim=np.delete(labels_trim,self.abs_rem_ind[i])
    for i in range(len(self.abs_rem_ind_test)-1,-1,-1):
        labels_trim_test=np.delete(labels_trim_test,self.abs_rem_ind_test[i])      


    for recording in range(len(last_layer_activity)):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity[recording][0][event] - last_layer_activity[recording][0][event-1]

        #We expand the number of columns in last_layer_activity[i][1] 
        #from 10 to 10+1, in order to include delta_t
        lstm_input = np.zeros((last_layer_activity[recording][1].shape[0], 11))
        lstm_input[:,:-1] = last_layer_activity[recording][1]
        lstm_input[:,10] = delta_t_recording
        data_train.append(lstm_input)

    data_train=np.array(data_train)      

    for recording in range(len(last_layer_activity_test)):
        #calculate delta t
        delta_t_recording = np.empty(last_layer_activity_test[recording][0].shape[0], dtype=int)
        delta_t_recording[0] = int(0)
        for event in range(1, len(delta_t_recording)):
            delta_t_recording[event] = last_layer_activity_test[recording][0][event] - last_layer_activity_test[recording][0][event-1]

        #We expand the number of columns in last_layer_activity[i][1] 
        #from 10 to 11, in order to include delta_t
        lstm_input = np.zeros((last_layer_activity_test[recording][1].shape[0], 11))
        lstm_input[:,:-1] = last_layer_activity_test[recording][1]
        lstm_input[:,10] = delta_t_recording
        data_test.append(lstm_input)

    data_test=np.array(data_test)  
    # Create the bins    
    activity_binned_train = []
    labels_bins_train = []
    sliding_amount=1
    
    for i in range(len(data_train)):
        if data_train[i].shape[0] >= bin_size: #if bin_size >= data_train length for that file
            activity_binned_train.append(data_train[i][0:bin_size, :])
            labels_bins_train.append(labels_trim[i])
            for j in range(bin_size - 1 + sliding_amount, data_train[i].shape[0], sliding_amount):
                activity_binned_train.append(data_train[i][j-bin_size:j, :])
                labels_bins_train.append(labels_trim[i])
        else: # if the last layer activity length for this file is less than the bin_size, we have to pad with zeros
            z = np.zeros([bin_size, data_train[i].shape[1]])
            z[:data_train[i].shape[0],:data_train[i].shape[1]] = data_train[i]
            activity_binned_train.append(z)
            labels_bins_train.append(labels_trim[i])

    labels_bins_train = np.array(labels_bins_train)


    activity_binned_test = []
    labels_bins_test = []
    for i in range(len(data_test)):
        if data_test[i].shape[0] >= bin_size: #if bin_size >= data_test length for that file
            activity_binned_test.append(data_test[i][0:bin_size, :])
            labels_bins_test.append(labels_trim_test[i])
            for j in range(bin_size - 1 + sliding_amount, data_test[i].shape[0], sliding_amount):
                activity_binned_test.append(data_test[i][j-bin_size:j, :])
                labels_bins_test.append(labels_trim_test[i])
        else: # if the last layer activity length for this file is less than the bin_size, we have to pad with zeros
            z = np.zeros([bin_size, data_test[i].shape[1]])
            z[:data_test[i].shape[0],:data_test[i].shape[1]] = data_test[i]
            activity_binned_test.append(z)
            labels_bins_test.append(labels_trim_test[i])

    labels_bins_test = np.array(labels_bins_test)                                                     

    self.cnn = create_CNN(learning_rate=learning_rate, width=n_bins, height=11)
    self.cnn.summary()

    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    self.tmp = [activity_binned_train, labels_bins_train, activity_binned_test, labels_bins_test]
    # fit model
    self.cnn.fit(activity_binned_train, labels_bins_train, 
      steps_per_epoch= X_train.shape[0]/batch_size,
      epochs=epochs, batch_size=batch_size, shuffle=True, 
      validation_data=(activity_binned_test, labels_bins_test),
      callbacks=[es])

    if self.exploring is True:
        print("Sonogram training ended, you can now access the trained network with the method .cnn")

    
    
    return




# Method for testing a cnn classification model 
# =============================================================================      
def cnn_classification_test(self, dataset_train, labels_train, dataset_test, labels_test, number_of_labels, batch_size, bin_size):
    
    """
    Method to test a previously-trained cnn, to a classification task, to compare the 
    feature extraction from the CNN with the feature representation 
    automatically extracted by HOTS
    
    Arguments:
        dataset_train (list) : list of lists with timestamps and addresses of all training files
        labels_train (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        dataset_test (list) : list of lists with timestamps and addresses of all testing files
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
        number_of_labels (int) : The total number of different labels that
                                 I am excepting in the dataset, (I know that i could
                                 max(labels) but, first, it's wasted computation, 
                                 second, the user should move his/her ass and eat 
                                 less donuts)
        batch_size (int) : mlp batchsize
        bin_size (int) : size (microseconds) of the window to integrate the spikes when 
                        calculating the sonogram

    """
    
    ### TRAIN ###
    
    train_images = []
    train_labels = []
    
    
    settings = pyNAVIS.MainSettings(num_channels=32, mono_stereo=0, bin_size=bin_size,on_off_both=0)
    
    for i in range(len(dataset_train)):
        spikes_file = pyNAVIS.SpikesFile(dataset_train[i][1], dataset_train[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        
        n_bins = 1000000//bin_size #number of bins that the sonogram should have (audios are 1 s long at maximum)
        
        if sonogram.shape[1] < n_bins:
            m_zeros = np.zeros((32, n_bins - sonogram.shape[1]))
            sonogram = np.concatenate((sonogram, m_zeros), axis=1)
        elif sonogram.shape[1] > n_bins:
            sonogram = sonogram[...,:-(sonogram.shape[1]-n_bins)]
        
        train_images.append(sonogram/np.max(sonogram)) #Normalize the sonogram between 0 and 1.
        train_labels.append(labels_train[i])
        
        
        
    
    y_train = np.asarray(train_labels)
    X_train = np.asarray(train_images)
    X_train = X_train.reshape((-1, settings.num_channels, n_bins, 1))
    
    
    
    
    ### TEST ###
    
    test_images = []
    test_labels = []    
    
    settings = pyNAVIS.MainSettings(num_channels=32, mono_stereo=0, bin_size=bin_size,on_off_both=0)
    
    for i in range(len(dataset_test)):
        spikes_file = pyNAVIS.SpikesFile(dataset_test[i][1], dataset_test[i][0]) #create SpikesFile object
        sonogram = pyNAVIS.Plots.sonogram(spikes_file, settings, return_data=True) #generate sonogram
        
        n_bins = 1000000//bin_size #number of bins that the sonogram should have (audios are 1 s long at maximum)
        
        if sonogram.shape[1] < n_bins:
            m_zeros = np.zeros((32, n_bins - sonogram.shape[1]))
            sonogram = np.concatenate((sonogram, m_zeros), axis=1)
        elif sonogram.shape[1] > n_bins:
            sonogram = sonogram[...,:-(sonogram.shape[1]-n_bins)]
        
        test_images.append(sonogram/np.max(sonogram)) #Normalize the sonogram between 0 and 1.
        test_labels.append(labels_test[i])
        
        
    y_test = np.asarray(test_labels)
    X_test = np.asarray(test_images)
    X_test = X_test.reshape((-1, settings.num_channels, n_bins, 1))
        
    
    
    
    predicted_labels_train = self.cnn.predict(X_train)
    predicted_labels_train = np.argmax(predicted_labels_train, axis=1)
    
    predicted_labels_test = self.cnn.predict(X_test)
    predicted_labels_test = np.argmax(predicted_labels_test, axis=1)
    
    
    print('Train Prediction rate is: '+str(accuracy_score(y_train, predicted_labels_train)*100)+'%') 
    print('Test Prediction rate is: '+str(accuracy_score(y_test, predicted_labels_test)*100)+'%') 

    
    return 