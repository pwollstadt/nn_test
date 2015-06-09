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



# main function
if __name__ == '__main__':
    
    #DATA INITIALIZATION
    gpuid = int(0)
    chunksize= 100000 #764400
    nchunkspergpu = int(10)
    kth = int(4)
    pointsdim = int(21)
    theiler = int(0)
    signallengthpergpu = nchunkspergpu * chunksize
    # set up problem iterations for averaging
    reps = 1 + 1 # +1 run in
    geometry = (reps, 1) # tuple!
    extimes=np.empty(geometry)
    
    for rr in range(reps):
        #Create an array of zeros for indexes and distances, and random array for pointset 
        indexes = np.zeros((kth, signallengthpergpu), dtype=np.int32)
        distances = np.zeros((kth, signallengthpergpu), dtype=np.float32)
        pointset = np.random.random((pointsdim, signallengthpergpu)).astype('float32')
        pointset2 = np.random.random((pointsdim, signallengthpergpu)).astype('float32')
        pointset3 = np.random.random((pointsdim, signallengthpergpu)).astype('float32')
        pointset4 =  np.random.random((pointsdim, signallengthpergpu)).astype('float32')
        #pointset = np.array( [(-12.1, 23.4, -20.6, 21.6, -8.5, 23.7, -10.1, 8.5), (5.3, -9.2, 8.2, -15.3, 15.1, -9.2,  5.5, -15.1) ], dtype=np.float32)
        queryset = pointset
        queryset2 = pointset2
        queryset3 = pointset3
        queryset4 = pointset4
        #Create an array of zeros for npointsrange, fill in pointset with a random vector, and start vecradius as an array of ones
        npointsrange2 = np.zeros((signallengthpergpu), dtype=np.int32)
        npointsrange3 = np.zeros((signallengthpergpu), dtype=np.int32)
        npointsrange4 = np.zeros((signallengthpergpu), dtype=np.int32)
        
        vecradius = 0.5 * np.ones((signallengthpergpu), dtype=np.float32)
           
        #GPU Execution
        start = time.time()
        # the opencl code works "in place", i.e distances, pointsets, etc. will be updated during the call
        # not returned explicitely
        correct_1 = clFindKnn(indexes, distances, pointset, queryset, kth, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        correct_2 = clFindRSAll(npointsrange2, pointset2, queryset2, vecradius, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        correct_3 = clFindRSAll(npointsrange3, pointset3, queryset3, vecradius, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        correct_4 = clFindRSAll(npointsrange4, pointset4, queryset4, vecradius, theiler, nchunkspergpu, pointsdim, signallengthpergpu, gpuid)
        end = time.time()
        extimes[rr,0] = end-start 
        meanextimes = np.average(extimes[1:]) 

       
    
    
    if correct_2 == 0:
        print ("GPU execution failed")
    else:
        print("Execution times:\n")
        print(extimes)
        print("average after run-in: \n")
        print(meanextimes)
#         print( pointset) 
#         print("Array of distances")
#         print(distances) 
#         print( "Array of index")
#         print(indexes)
