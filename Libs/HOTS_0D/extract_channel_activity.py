import numpy as np

def extract_channel_activity(spikes_file, channel):
    spikes_per_channel_ts = []
    for i in range(len(spikes_file.timestamps)):
        if spikes_file.addresses[i] == channel:
            spikes_per_channel_ts.append(spikes_file.timestamps[i])
    return spikes_per_channel_ts