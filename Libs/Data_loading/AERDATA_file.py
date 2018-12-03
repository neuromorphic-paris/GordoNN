#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dic 2 13:51:42 2018

@author: pedro

This file contains AERData_file class, used in the repo to manage the recordings.
Each object is composed by two lists : the adresses and the related timestampes
for each event extracted from a single file.
The addressed are composed by the channel id and the polarity.
In the case of a 4 channeled cochlea the total adresses will be 8,
with even ids for ON events and odd ids OFF off events,
meaning that the complete channel actvitity for let's say, channel 0, 
can be seen looking at address 0 for ON events and at address 1 for OFF events

Notice that timestamps it is going to be a sorted list from smaller to higher 
values representing each event timestamp in microseconds
 
"""
class AERDATA_file(object):
    address = []
    timestamp = []

    def __init__(self, addresses = [], timestamps = []):
        self.addresses = addresses
        self.timestamps = timestamps
