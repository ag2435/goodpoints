"""Squared-loss Gaussian kernel functionality.

Cython implementation of functions involving squared Gaussian kernel evaluation.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Squared-loss Gaussian Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double loss_gaussian_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil:
    """
    Base Gaussian kernel:
        k(X1, X2) = exp(-0.5 * ||X1-X2||_2^2/lam_sqd[j]) 
    Squared-loss Gaussian kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      k_params: array of (sig_sqd, product), where
        sig_sqd: kernel bandwidth squared
        product: number of dimensions to use for the product kernel 
            (counting backwards from the last dimension)
    """
    
    cdef double sig_sqd = k_params[0]
    cdef long product = <long>(k_params[1]) #1 if use product kernel, 0 otherwise
    cdef long d = X1.shape[0]
    
    cdef long j
    cdef double arg1, arg2, result
    
    # Compute the squared Euclidean distance between X1 and X2
    arg1 = 0
    for j in range(d-product):
        arg1 += (X1[j]-X2[j])**2
    
    # Compute the base squared Gaussian kernel
    result = exp(-0.5 * arg1 / sig_sqd)

    if product:
        # compute linear kernel
        arg2 = 0
        for j in range(d-product, d):
            arg2 += X1[j]*X2[j]
        result = result**2 + result * arg2

    return(result)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double loss_gaussian_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil:
    """
    Base Gaussian kernel:
        k(X1, X2) = exp(-0.5 * ||X1-X2||_2^2/lam_sqd[j]) 
    Squared-loss Gaussian kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between X1 and itself
    
    Args:
      X1: 1D array representing a data point
      k_params: array of (sig_sqd, product), where
        sig_sqd: kernel bandwidth squared
        product: number of dimensions to use for the product kernel 
            (counting backwards from the last dimension)
    """
    cdef long product = <long>(k_params[1]) #1 if use product kernel, 0 otherwise
    cdef long d = X1.shape[0]

    cdef double arg, result 
    
    result = 1

    if product:
        # Compute the linear kernel
        arg = 0
        for j in range(d-product, d):
            arg += X1[j]**2
        result = result**2 + result * arg

    return(result)