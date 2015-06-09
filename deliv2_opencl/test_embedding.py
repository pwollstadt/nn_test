import numpy as np # use numpy functions with np.function syntax
import time
from scipy.stats.mstats import zscore
from scipy.special import digamma

#DATA INITIALIZATION
gpuid = int(0)
numObservations = 10000
chunksize = numObservations #764400
nchunkspergpu = int(1)
kth = int(4)
#pointsdim = int(21)
theiler = int(0)
signallengthpergpu = nchunkspergpu * chunksize
# set up problem iterations for averaging
reps = 1 + 1 # +1 run in
extimes=np.empty(reps)

covariance = 0.4;
sourceArray = np.random.randn(numObservations).astype('float32')
destArray   = covariance*sourceArray[0:numObservations-1] + (1-covariance)*np.random.randn(numObservations - 1).astype('float32')
destArray = np.insert(destArray, 0, 0)

sourceArrayZ = zscore(sourceArray)
destArrayZ = zscore(destArray)

#embedding parameters
dim = 5
tau = 1
u   = 1
chunksize = numObservations - (dim-1) * tau - u

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

pointset_p2 = np.concatenate((pointset_p.reshape(pointset_p.shape[0],1), pointset_2),axis = 1)
pointset_21 = np.concatenate((pointset_2, pointset_1),axis = 1)
pointset_p21 = np.concatenate((pointset_p, pointset_21),axis = 1)
end_embedding = time.time()