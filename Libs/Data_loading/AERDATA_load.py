#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dic 2 13:51:42 2018

@author: pedro, marco

This file contains AERDATA_load function, used to read an AERDATA file and
create an AERDATA_file as result.

"""

import struct
import math
import gc


def AERDATA_load(path, address_size=2, use_all_addr = False):

    unpack_param = ">H"
    timestep = 0.2
    
    if address_size == 2:
        unpack_param = ">H"
        timestep = 0.2
    elif address_size == 4:
        unpack_param = ">L"
        timestep = 1.0
    else:
        print("[loadAEDAT] > SettingsError: Only address sizes implemented are 2 and 4 bytes")

    with open(path, 'rb') as f:
        ## Check header ##
        p = 0
        lt = f.readline()
        while lt and lt[0] == ord("#"):
            p += len(lt)
            lt = f.readline()
        f.seek(p)

        f.seek(0, 2)
        eof = f.tell()

        num_events = math.floor((eof-p)/(address_size + 4))

        f.seek(p)

        events = [0] * int(num_events)
        timestamps = [0] * int(num_events)

        ## Read file ##
        i = 0
        try:
            while 1:
                buff = f.read(address_size)
                x = struct.unpack(unpack_param, buff)[0]
                events[i] = int(x)

                buff = f.read(4)
                x = struct.unpack('>L', buff)[0]
                timestamps[i] = int(x * timestep)


                # If I read an event from the off address and use_all_addr is off
                # I stop the counter from advancing, the result will be evntually
                # overwritten or discarded in the last check before returning.
                if use_all_addr is False and events[i] % 2 != 0:
                    continue
                if use_all_addr is False:
                    events[i]=events[i]//2
                i += 1
            f.close()
            gc.collect()
        except Exception as inst:
            f.close()
            gc.collect()
            pass

        results_events = events[:i]
        results_timestamps = timestamps[:i]
    
    return results_events, results_timestamps

