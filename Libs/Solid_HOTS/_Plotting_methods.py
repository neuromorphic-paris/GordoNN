"""
@author: marcorax

This file contains Solid_HOTS_Net methods used only for printing results and 
state off the network
 
"""

# General purpouse libraries
import numpy as np 
import matplotlib.pyplot as plt

# Homemade Fresh Libraries like Grandma does
from ._General_Func import local_tv_generator, cross_tv_generator


# =============================================================================
def plt_surfaces_vs_reconstructions(self, file, layer=0, test=False):
    """
    Method plotting an all the time surfaces of file vs reconstuction
    to test the quality of the autoencoders
    Arguments : 
        file (int) : The index for the selected file
        layer (int) : The index of HOTS network you want to test
        test (bool) : Decide if use test or train dataset
    """
    # Get the data
    if test:
        data_lc = self.layer_dataset_test
        data = self.net_0D_response_test
    else:
        data_lc = self.layer_dataset
        data = self.net_0D_response
    
    if layer == 0:
        # Build the original local time surfaces
        results = local_tv_generator(file, data_lc[0],
                                        self.polarities[layer], self.taus_T[layer],
                                        self.local_surface_length[layer], False,
                                        self.activity_th, self.spacing_local_T[layer])
    # else:
    #     # Build the original local time surfaces
    #     results = local_tv_generator_rate(file, data_lc[layer],
    #                                     self.polarities[layer], self.taus_T[layer],
    #                                     self.local_surface_length[layer], False,
    #                                     self.activity_th, self.spacing_local_T[layer])      
    
    original_lc_surfaces=results[1]
            
    # Reconstructing the local time surfaces
    encoded_lc_surfaces=self.sub_t[layer].predict(np.asarray(original_lc_surfaces))
    reconstr_lc_surfaces=self.sub_t[layer].cluster_centers_[encoded_lc_surfaces]
    
    # Build the original local time surfaces
    results = recording_surface_generator(file, data[layer],self.polarities[layer],
                                          self.features_number[layer],
                                          self.taus_2D[layer],False)
    
    original_surfaces=results[1]
    # Reconstructing the local time surfaces
    encoded_surfaces=self.sub_2D[layer].predict(original_surfaces)
    reconstr_surfaces=self.sub_2D[layer].cluster_centers_[encoded_surfaces]
    
    # Plotting
    plt.figure()
    plt.imshow(original_lc_surfaces.astype('float32'))
    plt.suptitle('Original local surfaces layer: '+ str(layer), fontsize=16)
    plt.yscale('log')
    plt.clim(0,1)
    plt.figure()
    plt.imshow(reconstr_lc_surfaces.astype('float32'))
    plt.yscale('log')
    plt.clim(0,1)
    plt.suptitle('Reconstructed local surfaces layer: '+ str(layer), fontsize=16)
    
    plt.figure()
    plt.imshow(np.transpose(original_surfaces.astype('float32')))
    plt.suptitle('Original surfaces layer: '+ str(layer), fontsize=16)
    # plt.yscale('log')
    plt.clim(0,1)
    plt.figure()
    plt.imshow(np.transpose(reconstr_surfaces.astype('float32')))
    # plt.yscale('log')
    plt.clim(0,1)
    plt.suptitle('Reconstructed surfaces layer: '+ str(layer), fontsize=16)
    
    return

# =============================================================================
def plt_loss_history(self,  layer=0):
    """
    Method plotting the evolution of reconstruction loss of network autoencoders
    Arguments : 
        layer (int) : The index for the layer you want to plot the loss history from
    """
    story = self.aenc_T[layer][0].history.history
    los = [story["loss"][ep] for ep in range(len(story["loss"]))]
    val_los = [story["val_loss"][ep] for ep in range(len(story["loss"]))]
    plt.figure() 
    plt.plot(los, label="los")
    plt.plot(val_los, label="val_los")
    plt.legend()
    plt.suptitle('Net loss hystory _T_ of layer: '+ str(layer), fontsize=16)
    
    story = self.aenc_2D[layer][0].history.history
    los = [story["loss"][ep] for ep in range(len(story["loss"]))]
    val_los = [story["val_loss"][ep] for ep in range(len(story["loss"]))]
    plt.figure() 
    plt.plot(los, label="los")
    plt.plot(val_los, label="val_los")
    plt.legend()
    plt.suptitle('Net loss hystory _2D_ of layer: '+ str(layer), fontsize=16)
    
    return

# =============================================================================
def plt_last_layer_activation(self, file, labels, labels_test, classes, test=False):
    """
    Method plotting the last layer activation of HOTS 
    Arguments : 
       file (int) : The index for the selected file
       labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
       labels_test (numpy array int) : array of integers (labels) of the dataset
                                    used for testing
       classes (list of strings) : List with names of each label to name the plots
       test (bool) : Decide if use test or train dataset

    """
    # Get the data
    if test:
        data=self.last_layer_activity_test[file]
        label = labels_test[file]
    else:
        data=self.last_layer_activity[file]
        label = labels[file]
        

    plt.figure()
    plt.plot(data[0],data[1])
    
    plt.suptitle('File: '+ str(file) +' Class: '+ classes[label], fontsize=16)
    plt.grid()
    
    return

#TODO SOLVE BUG FOR 0T PLOT FOR LAYER>1
# =============================================================================
def plt_reverse_activation(self, file, layer, sublayer, labels, labels_test,
                           classes, test=False):
    """
    Method plotting reverse activation (activity of a layer per event of a recording)
    This is good to have a visual reference on what the network is sensitive to.
    
    You should expect n+2 plots where n is the dimensionality of the bottleneck
    layer of the selected sublayer. The first plot is showing the original dataset
    The second is a heatmap of the same events obtained applying a 1-norm on 
    the local surfaces built at the first layer (layer 0), is indicative 
    of the data injected to the net.
    The other n plots are the output value of the selected sublayer printed 
    in the position of the input event generating the response (Bear in mind 
    that the effective number of points might be lower given spacing or other 
    discarding techinques reducing the number of events per layer)
    
    Arguments : 
        file (int) : The index for the selected file
        layer (int) : The index of HOTS network you want to test
        sublayer (int, either 0 or 1) : Index of the selected sublayer
        labels (numpy array int) : array of integers (labels) of the dataset
                                    used for training
        labels_test (numpy array int) : array of integers (labels) of the dataset
                                        used for testing
        classes (list of strings) : List with names of each label to name the plots
        test (bool) : Decide if use test or train dataset

    """
    # Get the data
    if test:
        original_data = self.layer_dataset_test[0]
        first_sublayer = self.net_local_response_test[0]
        file_class = classes[labels_test[file]]
        if sublayer :
            sublayer_data = self.net_cross_response_test[layer] 
            which_sublayer="2D"
        else:
            sublayer_data = self.net_local_response_test[layer]
            which_sublayer="0T"
    else:
        original_data=self.layer_dataset[0]
        first_sublayer = self.net_local_response[0]        
        file_class = classes[labels[file]]
        if sublayer :
            sublayer_data = self.net_cross_response[layer]
            which_sublayer="2D"
        else:
            sublayer_data = self.net_local_response[layer]
            which_sublayer="0T"


   

    # First print the original recording
    plt.figure()
    plt.suptitle('Original file: '+ str(file) +' Class: '+ file_class, fontsize=16)
    plt.scatter(original_data[file][0], original_data[file][1], s=1)
    xlims = plt.xlim()
    ylims = plt.ylim()
    lcs = local_tv_generator(original_data[file], self.polarities[0],\
                                    self.taus_T[0],\
                                    self.local_surface_length[0])
    # Plot heatmap
    plt.figure()
    plt.suptitle('Heatmap file: '+ str(file) +' Class: '+ file_class, fontsize=16)
    image=plt.scatter(original_data[file][0],original_data[file][1],
                      c=np.sum(lcs,1), s=0.1)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.colorbar(image) 
    
    # To plot the sublayer data on the points of original data i need to find
    # the original dataset values comparing timestamps
    out = first_sublayer[file] # output data of a selected recording
    orig = original_data[file] # original data of a single recording
    # Get the abs index of the original events corresponding to the ones of the out
    # data (checking for same timestamp out[0][indx]==orig[0] and channel out[1][indx]==orig[1])
    abs_indx = [np.where(((out[0][indx]==orig[0])*(out[1][indx]==orig[1])==1))[0][0] for indx in range(len(out[0]))]
    
    # checking if the selected layer is anything than the first sublayer
    if layer or sublayer: 
        out = sublayer_data[file]
        orig_timestamps = orig[0][abs_indx]
        # It might happen that two events might have the same timestamp while 
        # being generated in different channels, to keep the algorithm simple 
        # I will only plot such points in the first available channel
        rel_indx=[np.where(out[0][indx]==orig_timestamps)[0][0] for indx in range(len(out[0]))] 
        abs_indx = [abs_indx[ind] for ind in rel_indx] ######TODO A LOT OF MULTIPLE PLOT HERE.SOLVE IT
    
    # Prepare the variables required for plotting           
    orig_timestamps = orig[0][abs_indx]
    orig_channels = orig[1][abs_indx]
    
    # Get the number of plots
    n_features = self.features_number[layer][sublayer]
    
    plt.figure()
    plt.suptitle('Layer: '+str(layer)+' '+which_sublayer+'-response File: '+ str(file) +' Class: '+ file_class, fontsize=16)
    
    if sublayer:    
        features = sublayer_data[file][1]
    else:
        features = sublayer_data[file][2]
        
    for i_feature in range(n_features):
        # Extract events by feature
        rel_indx = [i for i, x in enumerate(features) if x == i_feature]
        timestamps = [orig_timestamps[i] for i in rel_indx]
        channels = [orig_channels[i] for i in rel_indx]
        image = plt.scatter(timestamps, channels, label='Feature '+str(i_feature))
    
    plt.legend()

    plt.xlim(xlims)
    plt.ylim(ylims)
    
    
    return

#TODO FINISH THIS METHOD
# =============================================================================
def plt_distribution_timestamps(self):
# #%%
# def mean_min_max_dt(timestamps):
#     dt = [(timestamps[i+1]-timestamps[i]) for i in range(len(timestamps)-1)]
#     return [np.mean(dt), min(dt), max(dt), dt]
# file = 0
# res=[]
# for channel in range(32):
#     original_timestamps, original_waveform, new_timestamps, new_waveform = first_layer_sampling_plot(Net.layer_dataset[0], Net.net_0D_response[0], file, channel) 
#     res.append(mean_min_max_dt(original_timestamps))
#     plt.figure()
#     plt.hist(res[-1][-1])
     
# #%% distributions of timestamps ALL
# file = 1
# res=[]
# modes=np.zeros(input_channels)
# means=np.zeros(input_channels)
# for file in range(len(dataset_train)):
#     for channel in range(input_channels):
#         original_timestamps,original_waveform = first_layer_sampling(dataset_train,  file, channel) 
#         if file == 0:
#             res.append([(original_timestamps[i+1]-original_timestamps[i]) for i in range(len(original_timestamps)-1)])
#         else:
#             res[channel]+=[(original_timestamps[i+1]-original_timestamps[i]) for i in range(len(original_timestamps)-1)]
            

# for channel in range(32):
#     modes[channel]=stats.mode(res[channel])[0]
#     means[channel]=np.mean(res[channel])
#     plt.figure()
#     plt.hist([res[channel][i] for i in range(len(res[channel])) if res[channel][i]>modes[channel]*20], bins=500, label="channel = "+str(channel), alpha=0.6)
#     plt.legend(loc='best')    
    

# #%% Compute the highest modes 
# res=[]
# modes=np.zeros([input_channels, len(dataset_train)])
# for file in range(len(dataset_train)):
#     for channel in range(input_channels):
#         original_timestamps,original_waveform = first_layer_sampling(dataset_train,  file, channel) 
#         tmp=[(original_timestamps[i+1]-original_timestamps[i]) for i in range(len(original_timestamps)-1)]
#         modes[channel, file]=stats.mode(tmp)[0]


# #%%
# last_l = Net.last_layer_activity
# def mean_min_max_dt(timestamps):
#     dt = [(timestamps[i+1]-timestamps[i]) for i in range(len(timestamps)-1)]
#     return [np.mean(dt), min(dt), max(dt)]
# file = 0
# res_file=[]
# min_mean = 0
# for file in range(120):
#     timestamps = last_l[file][0]
#     timestamps = [i for n, i in enumerate(timestamps) if i not in timestamps[:n]] # Strangely enough some events have the same timestamp :O
#     res_file.append(mean_min_max_dt(timestamps))
#     min_mean += res_file[-1][1] 
# min_mean=min_mean/120
    return

    
