#/home/mwibral/py34env/python3.4
'''
    *
    * testKNN_call.py
    *
    *  Created on: 01/04/2014
    *      Author: cgongut / wibral
    *
'''
from clKnnLibrary import * # imports flat names from the library

import numpy as np # use numpy functions with np.function syntax
import time
from scipy.stats.mstats import zscore
from scipy.special import digamma


# main function
def compare_openCL(chunksize, dim):
    
    #DATA INITIALIZATION
    gpuid = int(0)
    #chunksize= 10000 #764400
    nchunks = int(1)
    kth = int(4)
    #pointsdim = int(21)
    pointsdim = int(dim)
    tau = 1
    u = 1
    theiler = int(0)
    signallengthpergpu = int(nchunks * chunksize)
    # set up problem iterations for averaging
    reps = 4 # +1 run in
    extimes=np.empty((reps))
    
    numObservations = chunksize + (dim-1) * tau + u
    covariance = 0.4;
    sourceArray = np.random.randn(numObservations).astype('float32')
    destArray   = covariance*sourceArray[0:numObservations-1] + (1-covariance)*np.random.randn(numObservations - 1).astype('float32')
    destArray = np.insert(destArray, 0, 0)
    
    sourceArrayZ = zscore(sourceArray)
    destArrayZ = zscore(destArray)
    
    start_embedding = time.time()
    #initialize embedded point sets
    pointset_2 = np.zeros([chunksize,dim])
    pointset_1 = np.zeros([chunksize,dim])
    pointset_p = np.zeros([chunksize])
    count = 0
    
    for firstPoint in range(chunksize):
        lastPoint  = firstPoint + dim*tau 
        pointset_1[count,:] = sourceArrayZ[firstPoint:lastPoint:tau]
        pointset_2[count,:] = destArrayZ[firstPoint+u-1:lastPoint+u-1:tau]
        pointset_p[count] = destArrayZ[lastPoint+u-1]
        count += 1
    
    pointset_p2 = np.concatenate((pointset_p.reshape(pointset_p.shape[0],1), pointset_2),axis = 1).astype('float32')
    pointset_21 = np.concatenate((pointset_2, pointset_1),axis = 1).astype('float32')
    pointset_p21 = np.concatenate((pointset_p.reshape(pointset_p.shape[0],1), pointset_21),axis = 1).astype('float32')
    pointset_2 = pointset_2.astype('float32')
    print(pointset_p21.shape)
    print(type(pointset_p21[0][0]))
    end_embedding = time.time()
    print("time for embedding: {0}".format(end_embedding - start_embedding))
    
    for rr in range(reps):
        #Create output arrays 
        indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
        distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)
        count_2 = np.zeros((signallengthpergpu), dtype=np.int32)
        count_p2 = np.zeros((signallengthpergpu), dtype=np.int32)
        count_21 = np.zeros((signallengthpergpu), dtype=np.int32)
        
        #GPU Execution
        start = time.time()
        # the opencl code works "in place", i.e distances, pointsets, etc. will be updated during the call
        # not returned explicitely
        err_knn = clFindKnn(indexes, distances, pointset_p21, pointset_p21, kth, theiler, nchunks, int(dim*2+1), signallengthpergpu, gpuid)
        radius = distances[:][kth-1]
        #print(distances)
        #print(type(radius[0]))
        correct_2 = clFindRSAll(count_2, pointset_2, pointset_2, radius, theiler, nchunks, int(dim), signallengthpergpu, gpuid)
        correct_3 = clFindRSAll(count_p2, pointset_p2, pointset_p2, radius, theiler, nchunks, int(dim+1), signallengthpergpu, gpuid)
        correct_4 = clFindRSAll(count_21, pointset_21, pointset_21, radius, theiler, nchunks, int(dim*2), signallengthpergpu, gpuid)
        end = time.time()
        extimes[rr] = end-start 
        print("---------------- finished repetition {0} of {1}".format(rr + 1, reps))
    
    meanextimes = np.average(extimes[1:]) 

    if (err_knn == 0):
        print ("GPU execution failed")
    else:
        print("\nExecution times:")
        print(extimes)
        print("average after run-in: {0} (d = {1}, no. points = {2})".format(meanextimes, dim, numObservations))
        
    return meanextimes

 
