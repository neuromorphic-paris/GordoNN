#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:23:15 2022

@author: marcorax93
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#%% 2D HOTS vs 0D HOTS simulation (ideal constant amplitude)

simulation_time = 500#ms
#sinusoid amplitude 
Baseline = 5
amp = Baseline*np.ones(simulation_time)
amp[100:300] = Baseline+5*np.sin(np.arange(100,300)*np.pi/100) # peaks at 200ms

#cochlea channels
freq = (np.array([20, 25, 31, 39, 49, 61, 77, 97, 121, 152, 191, 240, 300, 377, 
                 472, 592, 742, 930, 1166, 1462, 1833, 2297, 2880, 3610, 4525,\
                 5672, 7110, 8912, 11171, 14002, 17551, 22000]))*1e-3
    
freq = freq[:14]#cut high freq channels to avoid problems with the nyquist frequency

n_freq = len(freq)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#Create sinusoids
y = amp*np.sin(2*np.pi*np.outer(freq,np.arange(0,simulation_time)))
ax1.imshow(y, interpolation='nearest', aspect='auto')
ax1.set_title("Input stimuli")
ax1.set_xlabel("Time ms")
ax1.set_ylabel("Channel index")

ax2.plot(amp)
ax2.set_title("Amplitude modulation")
ax2.set_xlabel("Time ms")
ax2.set_ylabel("Amplitude")

#Create Events
timestamps = []
for t_bin in range(simulation_time):
    for channel in range(n_freq):
        if t_bin == 0:
            timestamps.append([])
        wave_ampl =  int(np.ceil(y[channel,t_bin]))
        timestamps[channel] += [t_bin+timestep/wave_ampl for timestep in range(wave_ampl)]
        
for channel in range(n_freq):
    ch_ts = np.array(timestamps[channel])
    timestamps[channel] = ch_ts #creating arrays out of lists
    ax3.scatter(ch_ts, channel*np.ones(len(ch_ts)),s=0.01)
ax3.set_title("Cochlea Scatter plot")
ax3.set_xlabel("Time ms")
ax3.set_ylabel("Channel index")    
ax3.set_ylim(ax3.get_ylim()[::-1])
    
n_events_channel = [ len(timestamps[channel]) for channel in range(n_freq) ]
ax4.plot(freq, n_events_channel)
ax4.set_title("N_events per channel")
ax4.set_xlabel("Channel freq")
ax4.set_ylabel("Number of events")    

#%% 1ms Time surface
#Create Time Surfaces
ts_decay=1#ms
tss = np.zeros([n_freq, len(timestamps[-1])])
tss[-1,:]=1
for event_ind, timestamp in enumerate(timestamps[-1]):
    for channel in range(n_freq-1):
        ch_ts = timestamps[channel]
        dt = (timestamp-ch_ts)*((timestamp-ch_ts)>=0)
        if sum(dt)==0:
            continue
        smallest_dt = np.min(dt[np.nonzero(dt)])
        tss[channel,event_ind] = np.exp(-(smallest_dt)/ts_decay)

fig, (axs) = plt.subplots(3, 2)
axs[0,0].imshow(tss, interpolation='nearest', aspect='auto')
axs[0,0].set_title("Time surface response 1ms decay")
axs[0,0].set_xlabel("Time ms")
axs[0,0].set_ylabel("Channel index")  

#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.transpose(tss))
ch_ts = timestamps[-1]
for channel in range(n_freq):
    axs[0,1].scatter(ch_ts, channel*np.ones(len(ch_ts)),s=5, c=kmeans.labels_)


axs[0,1].set_title("Clusters Scatter plot")
axs[0,1].set_xlabel("Time ms")
axs[0,1].set_ylabel("Channel index")    
axs[0,1].set_ylim(axs[0,1].get_ylim()[::-1])
    

#%% 200ms Time surface
#Create Time Surfaces
ts_decay=200#ms
tss = np.zeros([n_freq, len(timestamps[-1])])
tss[-1,:]=1
for event_ind, timestamp in enumerate(timestamps[-1]):
    for channel in range(n_freq-1):
        ch_ts = timestamps[channel]
        dt = (timestamp-ch_ts)*((timestamp-ch_ts)>=0)
        if sum(dt)==0:
            continue
        smallest_dt = np.min(dt[np.nonzero(dt)])
        tss[channel,event_ind] = np.exp(-(smallest_dt)/ts_decay)

axs[1,0].imshow(tss, interpolation='nearest', aspect='auto')
axs[1,0].set_title("Time surface response 200ms decay")
axs[1,0].set_xlabel("Time ms")
axs[1,0].set_ylabel("Channel index")  


#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.transpose(tss))
ch_ts = timestamps[-1]
for channel in range(n_freq):
    axs[1,1].scatter(ch_ts, channel*np.ones(len(ch_ts)),s=5, c=kmeans.labels_)


axs[1,1].set_title("Clusters Scatter plot")
axs[1,1].set_xlabel("Time ms")
axs[1,1].set_ylabel("Channel index")    
axs[1,1].set_ylim(axs[1,1].get_ylim()[::-1])
    

#%% 200ms Time Vectors
#Create Time Vectors
tv_decay=200#ms
tv_l = 100# Length of the timevector

for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    n_tv = int(n_ev/tv_l)
    ch_ts = ch_ts[:n_tv*tv_l]
    ch_contx = np.reshape(ch_ts,[n_tv,tv_l])
    tv_timestamp = ch_contx[:,-1]
    dt = np.transpose(tv_timestamp-np.transpose(ch_contx))
    tv = (np.exp(-(dt)/tv_decay))
    if channel == 0:
        tvs = tv
    else:
        tvs = np.concatenate([tvs,tv])
    ch_contx = ch_contx.flatten()
    axs[2,0].scatter(ch_contx, channel*np.ones(len(ch_ts)),s=5, c=tv.flatten())


axs[2,0].set_title("Time vectors")
axs[2,0].set_xlabel("Time ms")
axs[2,0].set_ylabel("Channel index")    
axs[2,0].set_ylim(axs[2,0].get_ylim()[::-1])

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
    axs[2,1].scatter(tv_timestamp, channel*np.ones(len(tv_timestamp)), s=5, c=ch_labels)
    label_count += n_labels
    
axs[2,1].set_title("Clusters Scatter plot")
axs[2,1].set_xlabel("Time ms")
axs[2,1].set_ylabel("Channel index")    
axs[2,1].set_ylim(axs[2,1].get_ylim()[::-1])
    

#%% Magnitude attenuation compensation

#cochlea channels
freq = (np.array([20, 25, 31, 39, 49, 61, 77, 97, 121, 152, 191, 240, 300, 377, 
                 472, 592, 742, 930, 1166, 1462, 1833, 2297, 2880, 3610, 4525,\
                 5672, 7110, 8912, 11171, 14002, 17551, 22000]))*1e-3
    
### Magnitude offset
actual_freq = freq*1e3

#Original model of magnitude offset
plt.figure()
orig_mag_model = np.linspace(2,9,32)

plt.plot(np.log(actual_freq), np.log(orig_mag_model), label="original approximation")


#Correct magnitude offset
#log-log line Y=mX + B 
#lin-lin curve  y = bx^m where B=log(b),Y=log(y),X=log(x)
# https://math.stackexchange.com/questions/3245738/finding-the-equation-of-a-straight-line-on-a-log-log-plot-given-two-points
y2=9
y1=2
x2=22e3
x1=20
m = np.log(y2/y1)/np.log(x2/x1)

B = np.log(y1)-m*np.log(x1)

new_mag_model_log = np.log(actual_freq)*m + B
new_mag_model = np.exp(new_mag_model_log)
plt.plot(np.log(actual_freq), np.log(new_mag_model), label="new approximation")
plt.legend()
plt.ylabel("log magnitude offs")
plt.xlabel("log frequency")
plt.title("Magnitude offset (it should be a straight line)")


#%% Cochlea simulation
simulation_time = 500#ms
#sinusoid amplitude 
Baseline = 5
amp = Baseline*np.ones(simulation_time)
amp[100:300] = Baseline+5*np.sin(np.arange(100,300)*np.pi/100) # peaks at 200ms

#cochlea channels    
freq = freq[:14]#cut high freq channels to avoid problems with the nyquist frequency
new_mag_model = new_mag_model[:14]/np.mean(new_mag_model[:14])#cut high freq channels to avoid problems with the nyquist frequency

y = amp*np.sin(2*np.pi*np.outer(freq,np.arange(0,simulation_time)))
y = np.transpose(new_mag_model*np.transpose(y))


n_freq = len(freq)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#Create sinusoids
ax1.imshow(y, interpolation='nearest', aspect='auto')
ax1.set_title("Input stimuli")
ax1.set_xlabel("Time ms")
ax1.set_ylabel("Channel index")

ax2.plot(amp)
ax2.set_title("Amplitude modulation")
ax2.set_xlabel("Time ms")
ax2.set_ylabel("Amplitude")

#Create Events
timestamps = []
for t_bin in range(simulation_time):
    for channel in range(n_freq):
        if t_bin == 0:
            timestamps.append([])
        wave_ampl =  int(np.ceil(y[channel,t_bin]))
        timestamps[channel] += [t_bin+timestep/wave_ampl for timestep in range(wave_ampl)]
        
for channel in range(n_freq):
    ch_ts = np.array(timestamps[channel])
    timestamps[channel] = ch_ts #creating arrays out of lists
    ax3.scatter(ch_ts, channel*np.ones(len(ch_ts)),s=0.01)
ax3.set_title("Cochlea Scatter plot")
ax3.set_xlabel("Time ms")
ax3.set_ylabel("Channel index")    
ax3.set_ylim(ax3.get_ylim()[::-1])

#%% 200ms Time Vectors (not compensated) CHECK HERE
#Create Time Vectors
tv_decay=200#ms
tv_l = 100# Length of the timevector
fig, (axs) = plt.subplots(2, 2)

#Downsampled tv calculation for plotting
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    n_tv = int(n_ev/tv_l)
    ch_ts = ch_ts[:n_tv*tv_l]
    ch_contx = np.reshape(ch_ts,[n_tv,tv_l])
    tv_timestamp = ch_contx[:,-1]
    dt = np.transpose(tv_timestamp-np.transpose(ch_contx))
    tv = (np.exp(-(dt)/tv_decay))
    if channel == 0:
        tvs = tv
    else:
        tvs = np.concatenate([tvs,tv])
    ch_contx = ch_contx.flatten()
    axs[0,0].scatter(ch_contx, channel*np.ones(len(ch_ts)),s=5, c=tv.flatten())

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



axs[0,0].set_title("Time vectors")
axs[0,0].set_xlabel("Time ms")
axs[0,0].set_ylabel("Channel index")    
axs[0,0].set_ylim(axs[0,0].get_ylim()[::-1])

#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(tvs)
label_count = 0
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    tv_timestamp = ch_ts[tv_l-1:]
    n_labels = len(tv_timestamp)
    ch_labels = kmeans.labels_[label_count:n_labels+label_count]
    axs[0,1].scatter(tv_timestamp, channel*np.ones(len(tv_timestamp)), s=50, c=ch_labels)
    label_count += n_labels
    
axs[0,1].set_title("Clusters Scatter plot")
axs[0,1].set_xlabel("Time ms")
axs[0,1].set_ylabel("Channel index")    
axs[0,1].set_ylim(axs[0,1].get_ylim()[::-1])

#% 200ms Time Vectors (compensated)
#Create Time Vectors
tv_decay=200#ms
tv_decays=tv_decay/new_mag_model#inverted magnitude model

tv_l = 100# Length of the timevector

#Downsampled tv calculation for plotting
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    n_tv = int(n_ev/tv_l)
    ch_ts = ch_ts[:n_tv*tv_l]
    ch_contx = np.reshape(ch_ts,[n_tv,tv_l])
    tv_timestamp = ch_contx[:,-1]
    dt = np.transpose(tv_timestamp-np.transpose(ch_contx))
    tv = (np.exp(-(dt)/tv_decays[channel]))
    if channel == 0:
        tvs = tv
    else:
        tvs = np.concatenate([tvs,tv])
    ch_contx = ch_contx.flatten()
    axs[1,0].scatter(ch_contx, channel*np.ones(len(ch_ts)),s=5, c=tv.flatten())

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

axs[1,0].set_title("Time vectors")
axs[1,0].set_xlabel("Time ms")
axs[1,0].set_ylabel("Channel index")    
axs[1,0].set_ylim(axs[1,0].get_ylim()[::-1])

#Clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(tvs)
label_count = 0
for channel in range(n_freq):
    ch_ts = timestamps[channel]
    n_ev = len(ch_ts)
    tv_timestamp = ch_ts[tv_l-1:]
    n_labels = len(tv_timestamp)
    ch_labels = kmeans.labels_[label_count:n_labels+label_count]
    axs[1,1].scatter(tv_timestamp, channel*np.ones(len(tv_timestamp)), s=50, c=ch_labels)
    label_count += n_labels



axs[1,1].set_title("Clusters Scatter plot")
axs[1,1].set_xlabel("Time ms")
axs[1,1].set_ylabel("Channel index")    
axs[1,1].set_ylim(axs[1,1].get_ylim()[::-1])
