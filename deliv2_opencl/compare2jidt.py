#/home/patriciaw/test_mpi/gpu_code/deliv2
'''
    *
    * compare2jidt.py
    *
    *  Created on: 12/11/2014
    *      Author: cgongut / wibral / pwollsta
    *
'''
from clKnnLibrary import * # imports flat names from the library

import numpy as np # use numpy functions with np.function syntax
from benchmark import compare_openCL

# the following are the values used by Joe:
points4benchmark = np.array([10000, 20000, 30000, 50000, 100000])
dims4benchmark = np.array([1, 2, 3, 5, 6, 8, 10])

ex_times = np.empty((points4benchmark.shape[0] * dims4benchmark.shape[0]))
count = 0

for n_points in points4benchmark:
    for dim in dims4benchmark:
        ex_times[count] = compare_openCL(n_points, dim)
        count += 1

np.save('benchmark.npy', ex_times)
np.savetxt('benchmark.txt', ex_times, delimiter=',', fmt='%f')