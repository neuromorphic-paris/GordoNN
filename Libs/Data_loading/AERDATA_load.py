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

# A simple function that reads events from AERDATA file
# =============================================================================
#    Args:
#        path(string): file name and its path
#        use_all_addr : if False all off events will be dropped, and the total
#                       addresses number will correspond to the number of
#                       channel of the cochlea
#    
#    Returns:
#        AERDATA_file: Object containing events extracted from the file
# =============================================================================

def AERDATA_load(path, use_all_addr = False):

    with open(path, 'rb') as f:
        ## Check header ##
        p = 0
        lt = f.readline()
        while lt and lt[0] == "#":
            p += len(lt)
            lt = f.readline()
        f.seek(p)

        f.seek(0, 2)
        eof = f.tell()

        num_events = math.floor((eof-p)/(2 + 4))

        f.seek(p)

        addresses = [0] * int(num_events)
        timestamps = [0] * int(num_events)
        ## Read file ##
        i = 0
        try:
            while 1:
                buff = f.read(2)      
                x = struct.unpack(">H", buff)[0]

                addresses[i] = int(x)

                buff = f.read(4)
                x = struct.unpack('>L', buff)[0]
                timestamps[i] = int(x * 0.2)
                # If I read an event from the off address and use_all_addr is off
                # I stop the counter from advancing, the result will be evntually
                # overwritten or discarded in the last check before returning.
                if use_all_addr is False and addresses[i] % 2 != 0:
                    continue
                if use_all_addr is False:
                    addresses[i]=addresses[i]//2
                i += 1
            f.close()
            gc.collect()
        except Exception as inst:
            #print(inst)
            f.close()
            gc.collect()
            pass
        result_addresses=addresses[:i]
        result_timestamps=timestamps[:i]
    return result_addresses, result_timestamps