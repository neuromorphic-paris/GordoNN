"""
@author: marcorax

This file contains the Solid_HOTS_Net class complementary functions that are used
inside the network that have no pratical use outside the context of the class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

 
"""

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, MaxPooling1D, Conv1D, Flatten, Activation


# =============================================================================
def events_from_activations_T(activations, events):
    """
    Function used to compute events out of simple activations (raw values)
    for a single recording
    Arguments :
        activations (list of list of floats): a list containing all latent values
                                              for a time surface
    Returns :
        events (nested list) : a list containing all events generated for the current
                               recording
    """
    out_timestamps = []
    out_polarities = []
    out_features = []
    out_rates = []
    for i in range(len(events[0])):
        t = events[0][i]
        for j,a_j in enumerate(activations[i]):
            out_timestamps.append(t)
            out_polarities.append(events[1][i])
            out_features.append(j)
            out_rates.append(a_j)



    out_timestamps = np.array(out_timestamps)
    out_polarities = np.array(out_polarities)
    out_features = np.array(out_features)
    out_rates = np.array(out_rates)
    
    return [out_timestamps, out_polarities, out_features, out_rates]

# The recording is selected outside, this is confusing a little
# =============================================================================
def events_from_activations_T_mod(activations, context_indices, events):
    """
    Function used to compute events out of simple activations (raw values)
    for a single recording
    Arguments :
        activations (list of list of floats): a list containing all latent values
                                              for a time surface
    Returns :
        events (nested list) : a list containing all events generated for the current
                               recording
    """
    out_timestamps = []
    out_polarities = []
    out_features = []
    out_rates = []
    count=0
    for idx in context_indices:
        t = events[0][idx]
        for j,a_j in enumerate(activations[count]):
            out_timestamps.append(t)
            out_polarities.append(events[1][idx])
            out_features.append(j)
            out_rates.append(a_j)
        count+=1

    out_timestamps = np.array(out_timestamps)
    out_polarities = np.array(out_polarities)
    out_features = np.array(out_features)
    out_rates = np.array(out_rates)
    
    return [out_timestamps, out_polarities, out_features, out_rates]

# The recording is selected outside, this is confusing a little
# =============================================================================
def events_from_activations_T_next_mod(activations, npolarities, context_indices, events):
    """
    Function used to compute events out of simple activations (raw values)
    for a single recording
    Arguments :
        activations (list of list of floats): a list containing all latent values
                                              for a time surface
    Returns :
        events (nested list) : a list containing all events generated for the current
                               recording
    """
    out_timestamps = []
    out_polarities = []
    out_features = []
    out_rates = []
    count=0
    for idx in context_indices:
        t = events[0][idx//npolarities]
        for j,a_j in enumerate(activations[count]):
            out_timestamps.append(t)
            out_polarities.append(idx%npolarities)
            out_features.append(j)
            out_rates.append(a_j)
        count+=1

    out_timestamps = np.array(out_timestamps)
    out_polarities = np.array(out_polarities)
    out_features = np.array(out_features)
    out_rates = np.array(out_rates)
    
    return [out_timestamps, out_polarities, out_features, out_rates]

# =============================================================================
def events_from_activations_2D(activations, events):
    """
    Function used to compute events out of simple activations (raw values)
    for a single recording
    Arguments :
        activations (list of list of floats): a list containing all latent values
                                              for a time surface
    Returns :
        events (nested list) : a list containing all events generated for the current
                               recording
    """
    out_timestamps = []
    out_polarities = []
    out_rates = []
    for i in range(len(events[0])):
        t = events[0][i]
        for j,a_j in enumerate(activations[i]):
            out_timestamps.append(t)
            out_polarities.append(j)
            out_rates.append(a_j)



    out_timestamps = np.array(out_timestamps)
    out_polarities = np.array(out_polarities)
    out_rates = np.array(out_rates)    

    return [out_timestamps, out_polarities, out_rates]

# =============================================================================
def create_mlp(input_size, hidden_size, output_size, learning_rate):
    """
    Function used to create a small mlp used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        mlp (keras model) : the freshly baked network
    """
    def relu_advanced(x):
        return keras.activations.relu(x, alpha=0.3)
    
    inputs = Input(shape=(input_size,), name='encoder_input')
    x = BatchNormalization()(inputs)
    x = Dense(hidden_size, activation='sigmoid')(x)
    outputs = Dense(output_size, activation='sigmoid')(x)
    
    
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    mlp.compile(optimizer=adam,
              loss='binary_crossentropy')
    
    return mlp


def create_lstm(timesteps, features, hidden_size, learning_rate):
    """
    Function used to create a small LSTM used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        lstm (keras model) : the freshly baked network
    """
    
    lstm = keras.models.Sequential()
    # lstm.add(keras.layers.Dense(30, activation='sigmoid'))
    # lstm.add(keras.layers.Dense(30, activation='sigmoid'))
    lstm.add(keras.layers.Masking(input_shape=(timesteps, features)))
    lstm.add(keras.layers.BatchNormalization())
    lstm.add(keras.layers.LSTM(hidden_size, dropout=0.1))
    

    #model.add(LSTM(4, input_shape=(bin_width, 11), return_sequences=True))
    #model.add(LSTM(4))
    # lstm.add(keras.layers.Dense(hidden_size//2, activation='relu'))
    lstm.add(keras.layers.Dense(1, activation='sigmoid'))
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    lstm.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return lstm    
    

def create_CNN(width, height, learning_rate):
    """
    Function used to create a small CNN used for classification purposes 
    Arguments :
        input_size (int) : size of the input layer
        hidden_size (int) : size of the hidden layer
        output_size (int) : size of the output layer
        learning_rate (int) : the learning rate for the optimization alg.
    Returns :
        cnn (keras model) : the freshly baked network
    """
    
    def relu_advanced(x): #Just in case we want to use it
        return keras.activations.relu(x, alpha=0.3)
    
    cnn = keras.models.Sequential()
    
    cnn.add(keras.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1), activation='relu', input_shape=(height, width, 1)))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None))
    
    cnn.add(keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1), activation='relu'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None))
    
    cnn.add(keras.layers.Flatten())
    
    cnn.add(keras.layers.Dense(100))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))
    
    cnn.add(keras.layers.Dense(2, activation='sigmoid'))


    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    cnn.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return cnn    


def create_autoencoder(original_dim, latent_dim, intermediate_dim, learning_rate, l1_norm_coeff, exploring):
    """
    Function used to create a small autoencoder used each layer of Solid HOTS
    Arguments :
        original_dim (int) : size of the input layer
        latent_dim (int) : size of the output layer
        intermediate_dim (int) : size of the hidden layer
        learning_rate (int) : the learning rate for the optimization alg.
        l1_norm_coeff (float) : coefficient used to sparse regularization of 
                                bottleneck layer
        Returns :
        aenc (keras model) : the freshly baked network
        encoder (keras model) : the freshly baked encoder
        decoder (keras model) : the freshly baked decoder
        
    """

    # network parameters
    input_shape = (original_dim, )
    reg = regularizers.l1(l1_norm_coeff)
    def l1_dim_norm(activities):
        return l1_norm_coeff*K.sum(K.abs(activities),axis=-1)#/latent_dim
    
    def relu_advanced(x):
        return keras.activations.relu(x, alpha=0.3)
    
    # Activation function, put here to enable quick changes for testing purposes
    # act = "sigmoid"
    act = relu_advanced
    
    ### Autoencoder model = encoder + decoder
        
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = BatchNormalization()(inputs)
    x = Dense(intermediate_dim, activation=act)(x)
    # x = Dense(intermediate_dim, activation=act)(x)
    
    
    encoded = Dense(latent_dim, name='encoded', activation="sigmoid", activity_regularizer=reg)(x)
    # Instantiate encoder model
    encoder = Model(inputs, encoded, name='encoder')
    if exploring==True:
        encoder.summary()

    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_inputs')
    x = Dense(intermediate_dim, activation=act)(latent_inputs)
    # x = Dense(intermediate_dim, activation=act)(x)

    outputs = Dense(original_dim, activation="sigmoid")(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    if exploring==True:
        decoder.summary()
    
    # instantiate autoencoder model
    outputs = decoder(encoder(inputs))
    aenc = Model(inputs, outputs, name='autoencoder_model')
    
    # define optimizer
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    # compiling network
    aenc.compile(optimizer=adam, loss='mse')
    
    return aenc, encoder, decoder



def local_surface_plot(layer_dataset, context_indices, all_contexts, layer):
    # HERE CHANNELS MEANS POLARITIES #TODO, FIX THE DEFINITION
    context_length = len(all_contexts[0])
    count=0
    if layer == 0:   
        mean_contexts = [np.zeros(context_length)] 
        var_contexts = [np.zeros(context_length)] 
        channel_counter = [0]
        for record in range(len(layer_dataset)):
            for event_index in context_indices[record]:
                channel=layer_dataset[record][1][event_index]
                if len(mean_contexts)<=channel:
                    var_contexts += [np.zeros(context_length) for i in range(channel+1-len(mean_contexts))] 
                    channel_counter.extend([0 for i in range(channel+1-len(mean_contexts))])
                    mean_contexts += [np.zeros(context_length) for i in range(channel+1-len(mean_contexts))] 
                mean_contexts[channel]+=all_contexts[count]
                channel_counter[channel]+=1
                count+=1
    else:
        channel_num = len(layer_dataset[0][1][0])
        mean_contexts = [np.zeros(context_length) for channel in range(channel_num)] 
        var_contexts = [np.zeros(context_length) for channel in range(channel_num)] 
        channel_counter = [0 for channel in range(channel_num)]
        for record in range(len(layer_dataset)):
            for event_index in context_indices[record]:
                channel=event_index%channel_num
                mean_contexts[channel]+=all_contexts[count]
                channel_counter[channel]+=1
                count+=1

        
    for channel in range(len(channel_counter)):
        mean_contexts[channel]=mean_contexts[channel]/channel_counter[channel]
    
    count=0
    if layer==0:        
        for record in range(len(layer_dataset)):
            for event_index in context_indices[record]:
                channel=layer_dataset[record][1][event_index]
                var_contexts[channel]+=(mean_contexts[channel]-all_contexts[count])**2
                count+=1
    else:
        for record in range(len(layer_dataset)):
            for event_index in context_indices[record]:
                channel=event_index%channel_num
                var_contexts[channel]+=(mean_contexts[channel]-all_contexts[count])**2
                count+=1
                
    for channel in range(len(channel_counter)):
        var_contexts[channel]=var_contexts[channel]/channel_counter[channel]    
    plt.figure()
    plt.title(" Mean local surfaces ")
    for channel in range(len(channel_counter)):
        plt.plot(mean_contexts[channel], color=(((len(channel_counter)-channel-1)/len(channel_counter),(channel+1)/len(channel_counter),0,1)), label = "Channel " +str(channel) )
    plt.legend()
    plt.figure()
    plt.title(" Context Variance ")
    for channel in range(len(channel_counter)):
        plt.plot(var_contexts[channel], color=(((len(channel_counter)-channel-1)/len(channel_counter),(channel+1)/len(channel_counter),0,1)), label = "Channel " +str(channel) )
    plt.legend()
    plt.figure()
    plt.title(" Context Abs ")
    for channel in range(len(channel_counter)):
        plt.plot(channel, sum(abs(mean_contexts[channel])), marker='o', color=((0,1,0,1)), label = "Channel " +str(channel) )
    plt.legend()
    plt.pause(5)

def first_layer_sampling_plot(layer_dataset, net_t_response, file_num, channel_num):
    original_timestamps = [layer_dataset[file_num][0][i] for i in range(len(layer_dataset[file_num][0])) if  layer_dataset[file_num][1][i]==channel_num]
    original_waveform = [1/(original_timestamps[i+1]-original_timestamps[i]) for i in range(len(original_timestamps)-1)]
    original_waveform.append(original_waveform[-1]) # Zeropad last layer
    new_timestamps = [net_t_response[file_num][0][i] for i in range(len(net_t_response[file_num][0])) if  net_t_response[file_num][1][i]==channel_num]
    new_timestamps = [i for n, i in enumerate(new_timestamps) if i not in new_timestamps[:n]]
    new_waveform = []
    for j in range(len(original_timestamps)):
        for i in range(len(new_timestamps)):
            if original_timestamps[j]==new_timestamps[i]:
                new_waveform.append(original_waveform[j] )
                break
    plt.figure()
    plt.plot(original_timestamps, original_waveform)
    plt.scatter(new_timestamps, new_waveform, c="r")
    plt.suptitle('Channel: '+ str(channel_num), fontsize=16)
    return original_timestamps, original_waveform, new_timestamps, new_waveform


def first_layer_sampling(layer_dataset, file_num, channel_num):
    original_timestamps = [layer_dataset[file_num][0][i] for i in range(len(layer_dataset[file_num][0])) if  layer_dataset[file_num][1][i]==channel_num]
    original_waveform = [1/(original_timestamps[i+1]-original_timestamps[i]) for i in range(len(original_timestamps)-1)]
    original_waveform.append(0) # Zeropad last layer
    return original_timestamps, original_waveform

def other_layers_sampling_plot(layer_dataset, net_t_response, file_num, channel_num):
    original_timestamps = [layer_dataset[file_num][0][i] for i in range(len(layer_dataset[file_num][0]))]
    original_waveform = [layer_dataset[file_num][1][i][channel_num] for i in range(len(original_timestamps)-1)]
    original_waveform.append(original_waveform[-1]) # Zeropad last layer
    new_timestamps = [net_t_response[file_num][0][i] for i in range(len(net_t_response[file_num][0])) if net_t_response[file_num][1][i]==channel_num]
    new_timestamps = [i for n, i in enumerate(new_timestamps) if i not in new_timestamps[:n]]
    new_waveform = []
    for i in range(len(new_timestamps)):
        for j in range(len(original_timestamps)):
            if original_timestamps[j]==new_timestamps[i]:
                new_waveform.append(original_waveform[j] )
                break
    plt.figure()
    plt.plot(original_timestamps, original_waveform)
    plt.scatter(new_timestamps, new_waveform, c="r")
    plt.suptitle('Channel: '+ str(channel_num), fontsize=16)
    return original_timestamps, original_waveform, new_timestamps, new_waveform

def surfaces_plot(all_surfaces,polarities,features):
    mean_surface = np.mean(all_surfaces,0)
    var_surface = np.zeros(polarities*features) 
    n_events = len(all_surfaces)
    for event in range(n_events):
        var_surface += (mean_surface - all_surfaces[event])**2/n_events
    plt.figure()
    plt.title(" Mean Surface ")
    plt.imshow(np.reshape(np.array(mean_surface, dtype=float),[polarities,features]))
    plt.figure()
    plt.title(" Surfaces Variance ")
    plt.imshow(np.reshape(np.array(var_surface, dtype=float),[polarities,features]))
    print(" Surfaces Abs: ")
    print(sum(abs(mean_surface)))
    plt.pause(5)
    
def recording_local_surface_generator(recording, dataset, polarities, taus_T, context_length,
                                exploring, activity_th, spacing):
    indxs = []
    local_surfaces=[]
    for polarity in range(polarities):
        indxs_channel = np.where(dataset[recording][1]==polarity)[0]
        n_contexts_per_polarity=len(np.arange(context_length-1,len(indxs_channel),spacing))
        contexts_polarity = np.zeros([n_contexts_per_polarity,context_length], dtype="float16")
        contexts_polarity_indices = np.zeros(n_contexts_per_polarity,dtype=int)
        count=0
        new_context = np.zeros(context_length)
        for i in range(context_length-1,len(indxs_channel),spacing):
            ind = indxs_channel[i]
            timestamp = dataset[recording][0][ind]
            timestamps = [dataset[recording][0][j] for j in indxs_channel[(i+1-context_length):i][::-1]]
            new_context[0] = 1         
            new_context[1:] = np.exp(-(timestamp-timestamps)/taus_T[polarity])
            if np.sum(new_context)>= activity_th :
                contexts_polarity[count] = new_context
                contexts_polarity_indices[count] = ind
                new_context=contexts_polarity[-1]
                count+=1
        contexts_polarity_indices=contexts_polarity_indices[np.sum(contexts_polarity,1) != 0]
        contexts_polarity=contexts_polarity[np.sum(contexts_polarity,1) != 0]
        indxs.append(contexts_polarity_indices)
        local_surfaces.append(contexts_polarity)   
    all_contexts_recording = np.zeros([len(dataset[recording][1]),context_length], dtype="float16")
    context_indices = np.zeros(len(dataset[recording][1]),dtype=int)
    for polarity in range(polarities):   
        for i,ind in enumerate(indxs[polarity]):
            all_contexts_recording[ind] = local_surfaces[polarity][i]
            context_indices[ind] = ind
    #remove the zero elements
    context_indices=context_indices[np.sum(all_contexts_recording,1) != 0]
    all_contexts_recording=all_contexts_recording[np.sum(all_contexts_recording,1) != 0]

    if exploring is True:
        print("\r","Local surfaces generation :", (recording+1)/len(dataset)*100,"%", end="")
    return(context_indices, all_contexts_recording)


def recording_local_surface_generator_rate(recording, dataset, polarities, taus_T, context_length,
                                exploring, activity_th, spacing):
    indxs = []
    local_surfaces=[]
    for polarity in range(polarities):
        n_contexts_per_polarity=len(dataset[recording][0])
        contexts_polarity = np.zeros([n_contexts_per_polarity,context_length], dtype="float16")
        contexts_polarity_indices = np.zeros(n_contexts_per_polarity,dtype=int)
        for ind in range(context_length-1,len(dataset[recording][0]),spacing):
            timestamp = dataset[recording][0][ind]
            timestamps = dataset[recording][0][(ind+1-context_length):ind+1][::-1]
            rates=dataset[recording][1][(ind+1-context_length):ind+1,polarity][::-1]
            new_context = rates*np.exp(-(timestamp-timestamps)/taus_T[polarity])
            if np.sum(new_context)>= activity_th :
                contexts_polarity[ind] = new_context
                contexts_polarity_indices[ind] = ind*(polarities) + polarity
        contexts_polarity_indices=contexts_polarity_indices[np.sum(contexts_polarity,1) != 0]
        contexts_polarity=contexts_polarity[np.sum(contexts_polarity,1) != 0]
        indxs.append(contexts_polarity_indices)
        local_surfaces.append(contexts_polarity)   
    all_contexts_recording = np.zeros([len(dataset[recording][0])*polarities,context_length], dtype="float16")
    context_indices = np.zeros(len(dataset[recording][1])*polarities, dtype=int)
    for polarity in range(polarities):   
        for i,ind in enumerate(indxs[polarity]):
            all_contexts_recording[ind] = local_surfaces[polarity][i]
            context_indices[ind] = ind
    #remove the zero elements
    context_indices=context_indices[np.sum(all_contexts_recording,1) != 0]
    all_contexts_recording=all_contexts_recording[np.sum(all_contexts_recording,1) != 0]
    
    if exploring is True:
        print("\r","Local surfaces generation :", (recording+1)/len(dataset)*100,"%", end="")
    return(context_indices, all_contexts_recording)
    

    
def recording_surface_generator(recording, dataset, polarities, features_number, 
                                taus_2D, exploring):   
    # 2D timesurface dimension
    ydim,xdim = [polarities, features_number[0]]
    # Per each event the first sublayer generates features_number[layer][0] new events
    # generating multiple surfaces at the same timestamp would be pointless
    reference_event_jump=features_number[0] 
    all_surfaces_recording = np.zeros([len(dataset[recording][0])//reference_event_jump,xdim*ydim], dtype="float16")
    context_indices = np.zeros([len(dataset[recording][0])//reference_event_jump], dtype="int")
    timestamps = np.zeros([ydim, xdim], dtype=int)
    rates = np.zeros([ydim, xdim], dtype="float16")
    timesurf = np.zeros([ydim, xdim], dtype="float16")

    count = 0 
    old_timestamp = dataset[recording][0][0]       
    for event_ind in range(0, len(dataset[recording][0]), reference_event_jump):   
        new_timestamp = dataset[recording][0][event_ind]                                   
        if new_timestamp != old_timestamp: # Important because after each layer we
                                       # potentially have len(features_number[layer][1])
                                       # for each event with the same timestamp
            all_surfaces_recording[count] = (timesurf).flatten() 
            context_indices[count] = event_ind
            count+=1 
        new_timestamp = dataset[recording][0][event_ind]                                   
        polarity = dataset[recording][1][event_ind]
        rates[polarity,:] = dataset[recording][3][event_ind:event_ind+features_number[0]]
        timestamps[polarity,:] = dataset[recording][0][event_ind:event_ind+features_number[0]]
        timesurf = rates*np.exp(-(new_timestamp-timestamps)/taus_2D)*(timestamps!=0)   
        old_timestamp=new_timestamp
   
    # I am also loading the last computed timesurface 
    all_surfaces_recording[count] = (timesurf).flatten() 
    context_indices[count] = event_ind
    count+=1   
    all_surfaces_recording=all_surfaces_recording[:count]
    context_indices=context_indices[:count]
    if exploring is True:
        print("\r","Surfaces generation :", (recording+1)/len(dataset)*100,"%", end="")
    return(context_indices, all_surfaces_recording)

# compute absolute removed indices 
def compute_abs_ind(rel_ind):
    absolute_indices=[]
    num_of_layers=len(rel_ind)
    layers_indices=np.arange(num_of_layers-1,0,-1)
    for layer in layers_indices:
        absolute_indices += rel_ind[layer]
        for j in range(len(absolute_indices)):
            i=0
            for i in range(len(rel_ind[layer-1])):
                if absolute_indices[j]>=rel_ind[layer-1][i]:
                    absolute_indices[j]+=1
                i+=1
    if not not rel_ind[0]:
        absolute_indices += rel_ind[0]
        
    absolute_indices=np.sort(absolute_indices).astype(int)
    return absolute_indices
    