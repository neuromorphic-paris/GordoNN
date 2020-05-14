"""
@author: marcorax

This file contains the Solid_HOTS_Net class methods that are used
to train and tests classifiers to assess the pattern recognition performed by 
HOTS
 
"""

# Computation libraries
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 

# Homemade Fresh Libraries like Grandma does (careful, it's still hot!)
from ._General_Func import create_mlp
from ._General_Func import create_lstm


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
def lstm_classification_train(self, labels_train, labels_test, number_of_labels, bin_width, 
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
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from 10 to 11, in order to include delta_t
        b = np.zeros((last_layer_activity[i][1].shape[0], 11))
        b[:,:-1] = last_layer_activity[i][1]
        b[:,10] = delta_t_i
        last_layer_activity[i][1] = b
    
    for i in range(len(last_layer_activity_test)):
        #calcualte delta t
        delta_t_i = np.empty(last_layer_activity_test[i][0].shape[0], dtype=int)
        delta_t_i[0] = int(0)
        for j in range(1, len(delta_t_i)):
            delta_t_i[j] = last_layer_activity_test[i][0][j] - last_layer_activity_test[i][0][j-1]
        
        #We expand the number of columns in last_layer_activity[i][1] 
        #from 10 to 11, in order to include delta_t
        b = np.zeros((last_layer_activity_test[i][1].shape[0], 11))
        b[:,:-1] = last_layer_activity_test[i][1]
        b[:,10] = delta_t_i
        last_layer_activity_test[i][1] = b
                                                     
                  
    # Create the bins    
    activity_binned_train = []
    labels_bins_train = []
    for i in range(len(last_layer_activity)):
        if last_layer_activity[i][1].shape[0] >= bin_width: #if bin_width >= last_layer_activity length for that file
            activity_binned_train.append(last_layer_activity[i][1][0:bin_width, :])
            labels_bins_train.append(labels_train[i])
            for j in range(bin_width - 1 + sliding_amount, last_layer_activity[i][1].shape[0], sliding_amount):
                activity_binned_train.append(last_layer_activity[i][1][j-bin_width:j, :])
                labels_bins_train.append(labels_train[i])
        else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
            z = np.zeros([bin_width, last_layer_activity[i][1].shape[1]])
            z[:last_layer_activity[i][1].shape[0],:last_layer_activity[i][1].shape[1]] = last_layer_activity[i][1]
            activity_binned_train.append(z)
            labels_bins_train.append(labels_train[i])
    
    labels_bins_train = np.array(labels_bins_train)


    activity_binned_test = []
    labels_bins_test = []
    for i in range(len(last_layer_activity_test)):
        if last_layer_activity_test[i][1].shape[0] >= bin_width: #if bin_width >= last_layer_activity length for that file
            activity_binned_test.append(last_layer_activity_test[i][1][0:bin_width, :])
            labels_bins_test.append(labels_test[i])
            for j in range(bin_width - 1 + sliding_amount, last_layer_activity_test[i][1].shape[0], sliding_amount):
                activity_binned_test.append(last_layer_activity_test[i][1][j-bin_width:j, :])
                labels_bins_test.append(labels_test[i])
        else: # if the last layer activity length for this file is less than the bin_width, we have to pad with zeros
            z = np.zeros([bin_width, last_layer_activity_test[i][1].shape[1]])
            z[:last_layer_activity_test[i][1].shape[0],:last_layer_activity_test[i][1].shape[1]] = last_layer_activity_test[i][1]
            activity_binned_test.append(z)
            labels_bins_test.append(labels_train[i])
    
    labels_bins_test = np.array(labels_bins_test)                                                     
                  
                    
                  
                                                     
    self.lstm = create_lstm(input_size=bin_width, hidden_size=units, learning_rate=learning_rate)
    self.lstm.summary()
    # Set early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    
    # fit model
    self.lstm.fit(np.array(activity_binned_train), labels_bins_train,
      epochs=epochs,
      batch_size=batch_size, shuffle=True, validation_data=(np.array(activity_binned_test), labels_bins_test),
      callbacks=[es], verbose=1)
        
    return 