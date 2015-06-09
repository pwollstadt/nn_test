#!/usr/bin/env python

'''
    *
    * matlab_to_pyOpenCL_RSAll.py
    *
    *  Created on: 01/10/2014
    *      Author: cgongut
    *
'''

from gpuKnnLibrary import *

import numpy as np
import time
import h5py

# main function
if __name__ == '__main__':
    
    f = h5py.File('RSAllData.mat','a')
    x = f["pointset"]
    pointset = np.array(x).astype('float32')
    x = f["radius"]
    vecradius = np.array(x).astype('float32')
    x = f["chunksize"]
    chunksize = int(x[0,0])
    x = f["nchunks"]
    nchunks = int(x[0,0])
    x = f["kth"]
    kth = int(x[0,0])
    x = f["thelier"]
    thelier = int(x[0,0])
    x = f["pointdim"]
    pointsdim = int(x[0,0])
    x = f["gpuid"]
    gpuid = int(x[0,0])

    #Create an array of zeros for indexes and distances
    signallength = nchunks * chunksize
    npointsrange = np.zeros((signallength), dtype=np.int32)
    queryset = pointset

    #GPU Execution
    start = time.time()
    correct = clFindRSAll(npointsrange, pointset, queryset, vecradius, thelier, nchunks, pointsdim, signallength, gpuid)
    end = time.time()
    
    if correct == 0:
        print ("GPU execution failed")
    else:
        print ("Saving distances and indexes from OpenCL to .mat file")
        f.create_dataset('npointsopencl', npointsrange.shape, npointsrange.dtype)
        f['npointsopencl'][:] = npointsrange

    f.close()

