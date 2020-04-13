"""
@author: marcorax

This file contains the Solid_HOTS_Net class methods that are used
to train and tests classifiers to assess the pattern recognition performed by 
HOTS
 
"""

# Computation libraries
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 

# Homemade Fresh Libraries like Grandma does
from ._General_Func import create_mlp


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


