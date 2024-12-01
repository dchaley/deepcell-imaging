# Adapted from: https://github.com/scikit-image/scikit-image/blob/3a82599073a9272cab0c370eccf678210a59f709/skimage/segmentation/heap_watershed.pxi
# Adapted from: https://github.com/scikit-image/scikit-image/blob/3a82599073a9272cab0c370eccf678210a59f709/skimage/segmentation/heap_general.pxi

cimport numpy as cnp


cdef struct Heapitem32:
    cnp.float32_t value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source

cdef struct Heapitem64:
    cnp.float64_t value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source

ctypedef fused Heapitem:
    Heapitem32
    Heapitem64

@cython.exceptval(check=False)
cdef inline int smaller(Heapitem *a, Heapitem *b) nogil:
    if a.value != b.value:
        return a.value < b.value
    return a.age < b.age


from libc.stdlib cimport free, malloc, realloc
from libc.stdint cimport uintptr_t


cdef struct Heap32:
    Py_ssize_t items
    Py_ssize_t space
    Heapitem32 *data
    Heapitem32 **ptrs

cdef struct Heap64:
    Py_ssize_t items
    Py_ssize_t space
    Heapitem64 *data
    Heapitem64 **ptrs

ctypedef fused Heap:
    Heap32
    Heap64

@cython.exceptval(check=False)
cdef inline void* heap_from_numpy2(Heap* _dummy) nogil:
    cdef Py_ssize_t k
    cdef Heap* heap
    if Heap is Heap32:
        heap = <Heap32 *> malloc(sizeof (Heap32))
    elif Heap is Heap64:
        heap = <Heap64 *> malloc(sizeof (Heap64))
    heap.items = 0
    heap.space = 1000
    if Heap is Heap64:
        heap.data = <Heapitem64 *> malloc(heap.space * sizeof(Heapitem64))
        heap.ptrs = <Heapitem64 **> malloc(heap.space * sizeof(Heapitem64 *))
    elif Heap is Heap32:
        heap.data = <Heapitem32 *> malloc(heap.space * sizeof(Heapitem32))
        heap.ptrs = <Heapitem32 **> malloc(heap.space * sizeof(Heapitem32 *))
    # else:
    #     raise ValueError("Invalid heap type")

    for k in range(heap.space):
        heap.ptrs[k] = heap.data + k
    return heap

@cython.exceptval(check=False)
cdef inline void heap_done(Heap* heap) nogil:
    free(heap.data)
    free(heap.ptrs)
    free(heap)

@cython.exceptval(check=False)
cdef inline void swap(Py_ssize_t a, Py_ssize_t b, Heap* h) nogil:
    h.ptrs[a], h.ptrs[b] = h.ptrs[b], h.ptrs[a]


######################################################
# heappop - inlined
#
# pop an element off the heap, maintaining heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
######################################################
@cython.exceptval(check=False)
cdef inline void heappop(Heap* heap, Heapitem* dest) nogil:

    cdef Py_ssize_t i, smallest, l, r # heap indices

    if Heap is Heap32 and Heapitem is Heapitem64:
        # error case, unreachable
        pass
    elif Heap is Heap64 and Heapitem is Heapitem32:
        # error case, unreachable
        pass
    else:
        #
        # Start by copying the first element to the destination
        #
        dest[0] = heap.ptrs[0][0]
        heap.items -= 1

        # if the heap is now empty, we can return, no need to fix heap.
        if heap.items == 0:
            return

        #
        # Move the last element in the heap to the first.
        #
        swap(0, heap.items, heap)

        #
        # Restore the heap invariant.
        #
        i = 0
        smallest = i
        while True:
            # loop invariant here: smallest == i

            # find smallest of (i, l, r), and swap it to i's position if necessary
            l = i * 2 + 1 #__left(i)
            r = i * 2 + 2 #__right(i)
            if l < heap.items:
                if smaller(heap.ptrs[l], heap.ptrs[i]):
                    smallest = l
                if r < heap.items and smaller(heap.ptrs[r], heap.ptrs[smallest]):
                    smallest = r
            else:
                # this is unnecessary, but trims 0.04 out of 0.85 seconds...
                break
            # the element at i is smaller than either of its children, heap
            # invariant restored.
            if smallest == i:
                break
            # swap
            swap(i, smallest, heap)
            i = smallest

##################################################
# heappush - inlined
#
# push the element onto the heap, maintaining the heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
##################################################
@cython.exceptval(check=False)
cdef inline int heappush(Heap* heap, Heapitem *new_elem) nogil:

    cdef Py_ssize_t child = heap.items
    cdef Py_ssize_t parent
    cdef Py_ssize_t k
    cdef Heapitem *new_data
    cdef Heapitem **new_ptr
    cdef uintptr_t original_data_ptr

    if Heap is Heap32 and Heapitem is Heapitem64:
        # error case, unreachable
        pass
    elif Heap is Heap64 and Heapitem is Heapitem32:
        # error case, unreachable
        pass
    else:

        # grow if necessary
        if heap.items == heap.space:
            heap.space = heap.space * 2

            # Original pointer to silence compiler warnings about use-after-free:
            original_data_ptr = <uintptr_t>heap.data
            new_data = <Heapitem *>realloc(<void *>heap.data,
                                           <Py_ssize_t>(heap.space * sizeof(Heapitem)))
            if not new_data:
                with gil:
                    raise MemoryError()

            # If necessary, correct all stored pointers:
            if new_data != heap.data:
                for k in range(heap.items):
                    # Calculate new pointers, `uintptr_t` avoids compiler warnings.
                    heap.ptrs[k] = <Heapitem *>(<uintptr_t>new_data + (
                            <uintptr_t>heap.ptrs[k] - original_data_ptr))
            heap.data = new_data

            new_ptrs = <Heapitem **>realloc(<void *>heap.ptrs,
                                            <Py_ssize_t>(heap.space * sizeof(Heapitem *)))
            if not new_ptrs:
                with gil:
                    raise MemoryError()
            heap.ptrs = new_ptrs

            # Initialize newly allocated pointer storage:
            for k in range(heap.items, heap.space):
                heap.ptrs[k] = new_data + k

        # insert new data at child
        heap.ptrs[child][0] = new_elem[0]
        heap.items += 1

        # restore heap invariant, all parents <= children
        while child > 0:
            parent = (child + 1) // 2 - 1 # __parent(i)

            if smaller(heap.ptrs[child], heap.ptrs[parent]):
                swap(parent, child, heap)
                child = parent
            else:
                break

        return 0