# Adapted from: https://github.com/scikit-image/scikit-image/blob/3a82599073a9272cab0c370eccf678210a59f709/skimage/segmentation/heap_watershed.pxi

cimport numpy as cnp


cdef struct Heapitem:
    cnp.float64_t value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source


cdef inline int smaller(Heapitem *a, Heapitem *b) nogil:
    if a.value != b.value:
        return a.value < b.value
    return a.age < b.age


include "heap_general.pxi"