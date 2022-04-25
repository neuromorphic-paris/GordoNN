#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:30:55 2022

@author: marcorax93
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Libs.Data_loading.AERDATA_load import AERDATA_load

# Create the dataset 
folder = "Data/white_noise_volumes/"
low_v_fname = "white_noise_10.aedat"
base_v_fname = "white_noise_50.aedat"
high_v_fname = "white_noise_100.aedat"

low_v_data = AERDATA_load(folder+low_v_fname, address_size=4)
base_v_data = AERDATA_load(folder+base_v_fname, address_size=4)
high_v_data = AERDATA_load(folder+high_v_fname, address_size=4)

low_v_data=[np.array(low_v_data[0]),np.array(low_v_data[1])]
base_v_data=[np.array(base_v_data[0]),np.array(base_v_data[1])]
high_v_data=[np.array(high_v_data[0]),np.array(high_v_data[1])]

# Set timestamps to relative
low_v_data[1][:] = np.array(low_v_data[1]) - low_v_data[1][0]
base_v_data[1][:] = np.array(base_v_data[1]) - base_v_data[1][0]
high_v_data[1][:] = np.array(high_v_data[1]) - high_v_data[1][0]

beg_recording = 1e6#ms

duration = 5e6+beg_recording#ms
beg_low = 1e6+beg_recording#ms
end_low = 2e6+beg_recording#ms
beg_high = 2e6+beg_recording#ms
end_high = 3e6+beg_recording#ms

data = [np.array([]),np.array([])]

# First section baseline
indx = base_v_data[1]<beg_low
data[0] = base_v_data[0][indx]
data[1] = base_v_data[1][indx]
# Second section low noise
indx = np.logical_and((low_v_data[1]<end_low), (low_v_data[1]>beg_low))
data[0] = np.concatenate([data[0], low_v_data[0][indx]])
data[1] = np.concatenate([data[1], low_v_data[1][indx]])
# Third section high noise
indx = np.logical_and((high_v_data[1]<end_high), (high_v_data[1]>beg_high))
data[0] = np.concatenate([data[0], high_v_data[0][indx]])
data[1] = np.concatenate([data[1], high_v_data[1][indx]])
# Fourth section baseline
indx = np.logical_and((base_v_data[1]<duration), (base_v_data[1]>end_high))
data[0] = np.concatenate([data[0], base_v_data[0][indx]])
data[1] = np.concatenate([data[1], base_v_data[1][indx]])


plt.figure()
plt.scatter(data[1] , data[0])