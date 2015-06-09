#!/usr/bin/env python

'''
    *
    * matlab_to_pyOpenCL_KNN.py
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
    
    f = h5py.File('KNNData.mat','a')
    x = f["pointset"]
    pointset = np.array(x).astype('float32')
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
    indexes = np.zeros((kth, signallength), dtype=np.int32)
    distances = np.zeros((kth, signallength), dtype=np.float32)
    queryset = pointset

    #GPU Execution
    start = time.time()
    bool = clFindKnn(indexes, distances, pointset, queryset, kth, thelier, nchunks, pointsdim, signallength, gpuid)
    end = time.time()
    
    if bool == 0:
        print ("GPU execution failed")
    else:
        print ("Saving distances and indexes from OpenCL to .mat file")
        f.create_dataset('distancesopencl', distances.shape, distances.dtype)
        f['distancesopencl'][:] = distances

        f.create_dataset('indexesopencl', indexes.shape, indexes.dtype)
        f['indexesopencl'][:] = indexes

    f.close()

