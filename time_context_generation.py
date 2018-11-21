import numpy as np
import math
import matplotlib.pyplot as plt

from AERDATAfile import *


def time_context_generation(spikes_files, tau, context_size):
    n_spikes = 0

    for i in range(len(spikes_files)):
        n_spikes += len(spikes_files[i])
    
    contexts = np.zeros((n_spikes, context_size)) #NOTE: this can be huge, but it's nice to preallocate
    cntx_id = 0

    for ind in range(len(spikes_files)):
        events_buffer = np.zeros(context_size) #NOTE: adding offset to extract the context easily
        spikes = spikes_files[ind]
        for ind_ev in range(len(spikes)):
            t_ev = spikes[ind_ev]
            for i in range(context_size):
                contexts[cntx_id, i] = math.exp(-(t_ev - events_buffer[i]) / tau)
            contexts[cntx_id, :] = np.roll(contexts[cntx_id,:], 1, axis = 0)
            contexts[cntx_id, 0] = 1
            events_buffer = np.roll(events_buffer, 1, axis = 0)
            events_buffer[0] = t_ev
            cntx_id += 1
        
    #plt.imshow(contexts, extent=[0, 1, 0, 1])
    #plt.colorbar()
    #plt.show()
    return contexts
