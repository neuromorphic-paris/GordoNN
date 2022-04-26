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
import pandas as pd


#%% CREATE THE DATASET - DATA LOADING
# Create the dataset 
folder = "Data/white_noise_volumes/"
low_v_fname = "white_noise_10.aedat"
base_v_fname = "white_noise_60.aedat"
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

n_freq=32

#%% CREATE THE DATASET - SEQUENCING THE RECORDING TOGETHER (base-volume, low-volume
#                                                           high-volume,base-volume)

beg_recording = 2e7#ms

duration = 5e6+beg_recording#ms
beg_low = 1e6+beg_recording#ms
end_low = 2e6+beg_recording#ms
beg_high = 2e6+beg_recording#ms
end_high = 3e6+beg_recording#ms

data = [np.array([]),np.array([])]

# First section baseline
indx = np.logical_and((base_v_data[1]<beg_low), (base_v_data[1]>beg_recording))
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

fig, (axs) = plt.subplots(3)
axs[0].scatter((data[1]-beg_recording)*1e-6, data[0], s=0.2)
axs[0].set_title("NOISE Test raw scatter-plot")
# axs[0].set_xlabel("Seconds")
axs[0].set_ylabel("Channel Index")
plt.xlabel("Time (seconds)")

#Change the dataset to channel centric representation
timestamps = []
for i_ch in range(n_freq):
    idx = data[0]==i_ch
    timestamps.append(np.asarray(data[1][idx], dtype='float64'))
    
    
#%% Clustering without volume compensation

#cochlea channels
freq = (np.array([20, 25, 31, 39, 49, 61, 77, 97, 121, 152, 191, 240, 300, 377, 
                 472, 592, 742, 930, 1166, 1462, 1833, 2297, 2880, 3610, 4525,\
                 5672, 7110, 8912, 11171, 14002, 17551, 22000]))[::-1]
    
#Create Time Vectors
tv_decay=5e2#ms
tv_l = 20# Length of the timevector

#Full tv calculation
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    n_tv = n_ev-tv_l+1
    times=np.transpose(np.tile(ch_ts,(n_ev,1)))
    dt = ch_ts-times
    # silent upper triangular elements of the vector elements outside the length
    # of the time vector
    dt[dt<0]=np.nan
    dt[np.triu(dt, k=tv_l)>0]=np.nan
    dt = np.transpose(dt[:,tv_l-1:])
    dt = dt[np.isfinite(dt)]
    dt = np.reshape(dt,[n_tv, tv_l])
    tv = (np.exp(-(dt)/tv_decay))
    if channel == 0:
        tvs = tv
    else:
        tvs = np.concatenate([tvs,tv])

#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(tvs)
label_count = 0
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    tv_timestamp = ch_ts[tv_l-1:]
    n_labels = len(tv_timestamp)
    ch_labels = kmeans.labels_[label_count:n_labels+label_count]
    axs[1].scatter((tv_timestamp-beg_recording)*1e-6, channel*np.ones(len(tv_timestamp)), s=0.8, c=ch_labels)
    label_count += n_labels


axs[1].set_title("Clustering without compensation")
axs[1].set_ylabel("Channel Index")    

#%% Clustering with volume compensation

df = pd.read_csv (r'whitenoise.csv')
mean_rate =np.asarray(df.columns[:], dtype=float)
mean_rate = mean_rate/mean_rate[0]

#cochlea channels
freq = (np.array([20, 25, 31, 39, 49, 61, 77, 97, 121, 152, 191, 240, 300, 377, 
                 472, 592, 742, 930, 1166, 1462, 1833, 2297, 2880, 3610, 4525,\
                 5672, 7110, 8912, 11171, 14002, 17551, 22000]))[::-1]
    
#Create Time Vectors
tv_decay=5e2#ms
tv_l = 20# Length of the timevector
tv_decays=tv_decay/mean_rate#inverted magnitude model

#Full tv calculation
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    n_tv = n_ev-tv_l+1
    times=np.transpose(np.tile(ch_ts,(n_ev,1)))
    dt = ch_ts-times
    # silent upper triangular elements of the vector elements outside the length
    # of the time vector
    dt[dt<0]=np.nan
    dt[np.triu(dt, k=tv_l)>0]=np.nan
    dt = np.transpose(dt[:,tv_l-1:])
    dt = dt[np.isfinite(dt)]
    dt = np.reshape(dt,[n_tv, tv_l])
    tv = (np.exp(-(dt)/tv_decays[channel]))
    if channel == 0:
        tvs = tv
    else:
        tvs = np.concatenate([tvs,tv])

#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(tvs)
label_count = 0
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    tv_timestamp = ch_ts[tv_l-1:]
    n_labels = len(tv_timestamp)
    ch_labels = kmeans.labels_[label_count:n_labels+label_count]
    axs[2].scatter((tv_timestamp-beg_recording)*1e-6, channel*np.ones(len(tv_timestamp)), s=0.8, c=ch_labels)
    label_count += n_labels


axs[2].set_title("Clustering with compensation")
axs[2].set_ylabel("Channel Index")    
