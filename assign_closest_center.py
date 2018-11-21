import numpy as np
import math
from scipy.spatial.distance import cdist

def assign_closest_center(spikes_files, centers, tau, context_size, metric_pdist):
    
    #closest_centers = np.zeros((len(spikes_files), []))
    closest_centers = []
    
    for ind in range(len(spikes_files)):
        events_buffer = np.zeros(context_size) #NOTE: adding offset to extract the context easily
        spikes = spikes_files[ind]
        ctx_spikes = np.zeros((len(spikes), context_size))
        cntx_id = 0
        
        for ind_ev in range(len(spikes)):
            t_ev = spikes[ind_ev]
            for i in range(context_size):
                ctx_spikes[cntx_id, i] = math.exp(-(t_ev - events_buffer[i]) / tau)
            ctx_spikes[cntx_id, :] = np.roll(ctx_spikes[cntx_id,:], 1, axis = 0)
            ctx_spikes[cntx_id, 0] = 1
            events_buffer = np.roll(events_buffer, 1, axis = 0)
            events_buffer[0] = t_ev
            cntx_id += 1
        
        closest_centers_file = np.argmin(cdist(ctx_spikes, centers), axis=1)
        closest_centers.append(closest_centers_file)
    return closest_centers