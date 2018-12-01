import glob
import struct
import math
import gc
from AERDATAfile import *

def loadAERDATA(path):

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
                addresses[i] = x

                buff = f.read(4)
                x = struct.unpack('>L', buff)[0]
                timestamps[i] = int(x * 0.2)

                i += 1
            f.close()
            gc.collect()
        except Exception as inst:
            f.close()
            gc.collect()
            pass
    return AERDATAfile(addresses, timestamps)