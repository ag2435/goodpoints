"""Squared loss Laplace kernel functionality.

Cython implementation of functions involving squared loss Laplace kernel evaluation.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Squared-loss Laplace Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double loss_laplace_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(- ||X1-X2||_2/ sig) 
    Squared-loss Laplace kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      k_params: array of (sig, product), where
        sig: kernel bandwidth
        product: number of dimensions to use for the product kernel 
            (counting backwards from the last dimension)
    """
    
    cdef double sig = k_params[0]
    cdef long product = <long>(k_params[1]) #1 if use product kernel, 0 otherwise
    cdef long d = X1.shape[0]
    
    cdef long j
    cdef double arg1, arg2, result
    
    # Compute the Euclidean distance between X1 and X2
    arg1 = 0
    for j in range(d-product):
        arg1 += (X1[j]-X2[j])**2
    arg1 = sqrt(arg1)
    
    # Compute the base Laplace kernel
    result = exp(-arg1 / sig)

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
cdef double loss_laplace_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(- ||X1-X2||_2/sig) 
    Squared-loss Laplace kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between X1 and itself
    
    Args:
      X1: 1D array representing a data point
      k_params: array of (sig, product), where
        sig: kernel bandwidth
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

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double prod_laplace_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(- ||X1-X2||_2/sig) 
    Squared-loss Laplace kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      k_params: array of (sig, product), where
        sig: kernel bandwidth
        product: number of dimensions to use for the product kernel 
            (counting backwards from the last dimension)
    """
    
    cdef double sig = k_params[0]
    cdef long product = <long>(k_params[1]) #1 if use product kernel, 0 otherwise
    cdef long d = X1.shape[0]
    
    cdef long j
    cdef double arg1, arg2, result
    
    # Compute the Euclidean distance between X1 and X2
    arg1 = 0
    for j in range(d-product):
        arg1 += (X1[j]-X2[j])**2
    arg1 = sqrt(arg1)
    
    # Compute the base Laplace kernel
    result = exp(-arg1 / sig)

    # compute linear kernel
    arg2 = 0
    for j in range(d-product, d):
        arg2 += X1[j]*X2[j]
    result = result * (1 + arg2)

    return(result)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double prod_laplace_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil:
    """
    Base Laplace kernel:
        k(X1, X2) = exp(-||X1-X2||_2/sig) 
    Squared-loss Laplace kernel:
        k_sq(X1, X2) = k(X1[:-product], X2[:-product])^2 
            + k(X1[:-product], X2[:-product]) * <X1[-product:], X2[-product:]>
    between X1 and itself
    
    Args:
      X1: 1D array representing a data point
      k_params: array of (sig, product), where
        sig: kernel bandwidth
        product: number of dimensions to use for the product kernel 
            (counting backwards from the last dimension)
    """
    cdef long product = <long>(k_params[1]) #1 if use product kernel, 0 otherwise
    cdef long d = X1.shape[0]

    cdef double arg, result 
    
    result = 1

    # Compute the linear kernel
    arg = 0
    for j in range(d-product, d):
        arg += X1[j]**2
    result = result * (1 + arg)

    return(result)