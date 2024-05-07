"""Cython declarations for Epanechnikov kernel functionality used by other files
"""
cdef double epanechnikov_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil

cdef double epanechnikov_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil