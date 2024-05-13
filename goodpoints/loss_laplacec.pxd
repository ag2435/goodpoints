"""Cython declarations for squared Laplace kernel functionality used by other files
"""
cdef double loss_laplace_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil

cdef double loss_laplace_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil

cdef double prod_laplace_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] k_params) noexcept nogil

cdef double prod_laplace_kernel_one_point(const double[:] X1,
                                      const double[:] k_params) noexcept nogil