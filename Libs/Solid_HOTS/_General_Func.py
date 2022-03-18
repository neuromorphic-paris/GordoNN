"""
@author: marcorax

This file contains the Solid_HOTS_Net class complementary functions that are used
inside the network that have no pratical use outside the context of the class
therefore is advised to import only the class itself and to use its method to perform
learning and classification

 
"""

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
import tensorflow.keras as keras
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, MaxPooling1D, Conv1D, Flatten, Activation


# =============================================================================
def events_from_local_activations(activations, events):
    """
    Function used to compute events out of local activations 
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
def events_from_cross_activations(features, events):
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
    for idx in context_indices:
        t = events[0][idx]
        out_timestamps.append(t)
        out_polarities.append(events[1][idx])

    out_timestamps = np.array(out_timestamps)
    out_polarities = np.array(out_polarities)
    out_features = np.array(out_features)
    
    return [out_timestamps, out_polarities, features]

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
    x = Dense(hidden_size)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)#0.3
    x = Dense(hidden_size)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)#0.7-0.3
    # x = Dense(hidden_size, activation='relu')(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(hidden_size, activation='relu')(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.9)(x)
    outputs = Dense(output_size, activation='sigmoid')(x)
    
    
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    mlp.compile(optimizer=adam,
             # loss='categorical_crossentropy', metrics=['accuracy'])
             loss='mean_squared_error', metrics=['accuracy'])
    
    return mlp

# =============================================================================
def create_simple_mlp(input_size, hidden_size, output_size, learning_rate):
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
    # x = BatchNormalization()(inputs)
    # x = Dense(hidden_size)(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = Dropout(0.7)(x)#0.3
    # x = Dense(hidden_size)(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = Dropout(0.7)(x)#0.7-0.3
    # x = Dense(hidden_size, activation='relu')(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(hidden_size, activation='relu')(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(hidden_size, activation='sigmoid')(x)
    # x = Dropout(0.9)(x)
    outputs = Dense(output_size, activation='sigmoid')(inputs)
    
    
    adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mlp = Model(inputs, outputs, name='mlp')
    mlp.compile(optimizer=adam,
             # loss='categorical_crossentropy', metrics=['accuracy'])
             loss='mean_squared_error', metrics=['accuracy'])
    
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
    lstm.add(keras.layers.LSTM(hidden_size, dropout=0.4))
    

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


def create_autoencoder(original_dim, latent_dim, intermediate_dim, learning_rate, l1_norm_coeff, verbose):
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
    if verbose==True:
        encoder.summary()

    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_inputs')
    x = Dense(intermediate_dim, activation=act)(latent_inputs)
    # x = Dense(intermediate_dim, activation=act)(x)

    outputs = Dense(original_dim, activation="sigmoid")(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    if verbose==True:
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
    # if layer == 0:   
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
    # else:
    #     channel_num = len(layer_dataset[0][1][0])
    #     mean_contexts = [np.zeros(context_length) for channel in range(channel_num)] 
    #     var_contexts = [np.zeros(context_length) for channel in range(channel_num)] 
    #     channel_counter = [0 for channel in range(channel_num)]
    #     for record in range(len(layer_dataset)):
    #         for event_index in context_indices[record]:
    #             channel=event_index%channel_num
    #             mean_contexts[channel]+=all_contexts[count]
    #             channel_counter[channel]+=1
    #             count+=1

        
    for channel in range(len(channel_counter)):
        mean_contexts[channel]=mean_contexts[channel]/channel_counter[channel]
    
    count=0
    # if layer==0:        
    for record in range(len(layer_dataset)):
        for event_index in context_indices[record]:
            channel=layer_dataset[record][1][event_index]
            var_contexts[channel]+=(mean_contexts[channel]-all_contexts[count])**2
            count+=1
    # else:
    #     for record in range(len(layer_dataset)):
    #         for event_index in context_indices[record]:
    #             channel=event_index%channel_num
    #             var_contexts[channel]+=(mean_contexts[channel]-all_contexts[count])**2
    #             count+=1
                
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
    
def local_tv_generator(recording_data, n_polarities, taus_T, context_length):
    """
    Function used to generate local time vectors.
    
    Arguments:
        
        recording_data: (list of 2 numpy 1D arrays) It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities: (int) the number of channels/polarities of the dataset.
        taus_T: (list of floats) the decays of the local time vector layer per each polarity.
        context_length: (int) the length of the local time vector 
                              (the length of its context).  
    
    Returns:
        
        local_tvs: (2D numpy array of floats) the timevectors generated for the 
                   recording where the dimensions are [i_event, timevector element]
    """
    
    n_events = len(recording_data[0])
    local_tvs=np.zeros([n_events,context_length])
    for polarity in range(n_polarities):
        indxs_channel = np.where(recording_data[1]==polarity)[0]
        new_local_tv = np.zeros(context_length)
        for i,ind in enumerate(indxs_channel):
            timestamp = recording_data[0][ind]
            # timestamps = [recording_data[0][j] for j in indxs_channel[(i+1-context_length):i][::-1]]
            context = recording_data[0][indxs_channel[(i+1-context_length):i][::-1]]
            new_local_tv[0] = 1         
            new_local_tv[1:1+len(context)] = np.exp(-(timestamp-context)/taus_T[polarity])
            local_tvs[ind] = new_local_tv

    return local_tvs

      
def cross_tv_generator(recording_data, n_polarities, features_number, tau_2D):
    """
    Function used to generate cross time vectors.
    
    Arguments:
        
        recording_data: (list of 2 numpy 1D arrays) It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities: (int) the number of channels/polarities of the dataset.
        tau_2D: (float) the single decay of the cross time vector layer.
        context_length: (int) the length of the cross time vector 
                              (the length of its context).  
    
    Returns:
        
        cross_tvs: (2D numpy array of floats) the timevectors generated for the 
                   recording where the dimensions are [i_event, 2D timevector element]
    """
    
    
    n_events = len(recording_data[0])
    
    # 2D timesurface dimension
    ydim,xdim = [n_polarities, features_number[0]]
    cross_tvs = np.zeros([len(recording_data[0]),xdim*ydim], dtype="float16")
    timestamps = np.zeros([ydim, xdim], dtype=int)
    
    if type(tau_2D)==np.ndarray:
        tau_2D_new = np.reshape(tau_2D, [ydim,xdim])
    else:
        tau_2D_new = tau_2D
    for event_ind in range(n_events):   
        new_timestamp = recording_data[0][event_ind]                                   
        polarity = recording_data[1][event_ind]
        feature = recording_data[2][event_ind]
        timestamps[polarity, feature] = recording_data[0][event_ind]
        # timesurf = np.exp(-(new_timestamp-timestamps)/tau_2D)*(timestamps!=0)   
        timesurf = np.exp(-(new_timestamp-timestamps)/tau_2D_new)*(timestamps!=0)   
        cross_tvs[event_ind] = (timesurf).flatten() 


    return cross_tvs

def cross_tv_generator_conv(recording_data, n_polarities, features_number, cross_size, tau_2D):
    """
    Function used to generate cross time vectors.
    
    Arguments:
        
        recording_data: (list of 2 numpy 1D arrays) It consists of the time stamps 
                        and polarity indeces for all the events of a single 
                        recording.
                        
        n_polarities: (int) the number of channels/polarities of the dataset.
        tau_2D: (float) the single decay of the cross time vector layer.
        context_length: (int) the length of the cross time vector 
                              (the length of its context).  
    
    Returns:
        
        cross_tvs: (2D numpy array of floats) the timevectors generated for the 
                   recording where the dimensions are [i_event, 2D timevector element]
    """
    
    
    n_events = len(recording_data[0])
    
    # 2D timesurface dimension
    ydim,xdim = [n_polarities, features_number[0]]
    cross_tvs_conv = np.zeros([len(recording_data[0]),cross_size], dtype="float16")
    zerp_off =  cross_size//2#zeropad offset
    timestamps = np.zeros([ydim+zerp_off*2, xdim], dtype=int) # ydim + 2* zeropad
    
    if type(tau_2D)==np.ndarray:
        tau_2D_new = np.reshape(tau_2D, [ydim,xdim])
    else:
        tau_2D_new = tau_2D
    for event_ind in range(n_events):   
        new_timestamp = recording_data[0][event_ind]                                   
        polarity = recording_data[1][event_ind]
        feature = recording_data[2][event_ind]
        timestamps[polarity+zerp_off, feature] = recording_data[0][event_ind]
        # timesurf = np.exp(-(new_timestamp-timestamps)/tau_2D)*(timestamps!=0)   
        timesurf = np.exp(-(new_timestamp-timestamps)/tau_2D_new)*(timestamps!=0)   
        timesurf = timesurf[(np.range(cross_size))+polarity]
        cross_tvs_conv[event_ind,:] = (timesurf).flatten() 


    return cross_tvs_conv


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
    