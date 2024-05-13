"""Laplace kernel functionality.

Cython implementation of functions involving Laplace kernel evaluation.
"""
import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport sqrt, log, exp, cos
from libc.stdlib cimport rand, RAND_MAX, srand
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdio cimport printf
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Laplace Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''


@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double laplace_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(- ||X1-X2||_2/ sig) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      k_params: array of (sig, product), where
        sig: kernel bandwidth
    """
    
    cdef double sig = k_params[0]
    cdef long d = X1.shape[0]
    
    cdef long j
    cdef double arg, result
    
    # Compute the Euclidean distance between X1 and X2
    arg = 0
    for j in range(d):
        arg += (X1[j]-X2[j])**2
    arg = sqrt(arg)
    
    # Compute the base Laplace kernel
    result = exp(-arg / sig)

    return(result)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double laplace_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(- ||X1-X2||_2/sig) 
    between X1 and itself
    
    Args:
      X1: 1D array representing a data point
      k_params: array of (sig, product), where
        sig: kernel bandwidth
    """
    return(1)
