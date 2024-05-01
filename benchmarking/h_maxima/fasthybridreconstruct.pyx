# cython:language_level=3

from collections import deque
import logging
import numpy as np
import timeit

cimport cython
from libc.stdint cimport uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t

# This fast-hybrid reconstruction algorithm supports the following data types.
# To add more, add to this list, and to the type cast in the main function.

# Production mode types
ctypedef fused image_dtype:
    int8_t
    uint8_t
    int16_t
    uint16_t
    int32_t
    uint32_t
    int64_t
    uint64_t
    float
    double

cpdef enum:
    METHOD_DILATION = 0
    METHOD_EROSION = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint8_t increment_index(
    Py_ssize_t* indices_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
) nogil:
    """Increment an index in N dimensions.

    Args:
        indices_ptr (Py_ssize_t*): the indices to increment
        dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        1 if the iteration is complete, 0 otherwise.
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        if indices_ptr[i] < dimensions_ptr[i] - 1:
            indices_ptr[i] += 1
            return 0
        else:
            indices_ptr[i] = 0
            if i == 0:
                return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint8_t decrement_index(
        Py_ssize_t* indices_ptr,
        Py_ssize_t* dimensions_ptr,
        Py_ssize_t num_dimensions,
) nogil:
    """Decrement an index in N dimensions.

    Args:
        indices_ptr (Py_ssize_t*): the indices to increment
        dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        1 if the iteration is complete, 0 otherwise.
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        if indices_ptr[i] > 0:
            indices_ptr[i] -= 1
            return 0
        else:
            indices_ptr[i] = dimensions_ptr[i] - 1
            if i == 0:
                return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t point_to_linear(
    Py_ssize_t* coord_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
) nogil:
    """Convert a point in N dimensions to a linear index.

    Args:
        coord_ptr (Py_ssize_t*): the point to convert
        dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        Py_ssize_t: the linear index
    """
    cdef Py_ssize_t linear = 0
    cdef Py_ssize_t multiplier = 1
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        linear += coord_ptr[i] * multiplier
        multiplier *= dimensions_ptr[i]
    return linear

# for testing; test_fast_hybrid.py
def point_to_linear_python(point, dimensions, num_dimensions):
    cdef Py_ssize_t* point_ptr = <Py_ssize_t*> <Py_ssize_t> point.ctypes.data
    cdef Py_ssize_t* dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> dimensions.ctypes.data
    cdef Py_ssize_t num_dims = <Py_ssize_t> num_dimensions
    return <long> point_to_linear(point_ptr, dimensions_ptr, num_dims)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t* linear_to_point(
    Py_ssize_t linear,
    Py_ssize_t* point_output_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
) nogil:
    """Convert a linear index to a point in N dimensions.

    Args:
        linear (Py_ssize_t): the linear index
        point_output_ptr (Py_ssize_t*): the point to write to
        dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        Py_ssize_t*: the point
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        point_output_ptr[i] = cython.cmod(linear, dimensions_ptr[i])
        linear = cython.cdiv(linear, dimensions_ptr[i])

# for testing; test_fast_hybrid.py
def linear_to_point_python(linear, dimensions, num_dimensions):
    point = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* point_output_ptr = <Py_ssize_t*> <Py_ssize_t> point.ctypes.data
    cdef Py_ssize_t* dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> dimensions.ctypes.data
    cdef Py_ssize_t num_dims = <Py_ssize_t> num_dimensions
    linear_to_point(linear, point_output_ptr, dimensions_ptr, num_dims)
    return point

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint8_t offset_coord(
    Py_ssize_t* start_coord,
    Py_ssize_t* footprint_center_offset,
    Py_ssize_t* offset_coord,
    Py_ssize_t* result_coord,
    Py_ssize_t sign,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
    Py_ssize_t* result_linear_index,
    uint8_t* at_center,
) nogil:
    """Offset (add) the starting point by the offset point minus a center point.

    If the sign is negative, the offset-center is subtracted instead.

    If any dimension is out of bounds, return false and skip assigning more coordinates.

    Otherwise, set all dimension coordinates, the linear index, and whether the point is at the center.

    Returns:
        uint8_t: 1 if the neighbor is in bounds, 0 otherwise.
    """
    cdef Py_ssize_t i
    result_linear_index[0] = 0
    cdef Py_ssize_t multiplier = 1
    at_center[0] = True
    for i in range(num_dimensions - 1, -1, -1):
        result_coord[i] = start_coord[i] + sign * (offset_coord[i] - footprint_center_offset[i])

        # Make sure the dimension is in bounds; return false if not.
        if result_coord[i] < 0 or result_coord[i] >= dimensions[i]:
            return 0

        result_linear_index[0] += result_coord[i] * multiplier
        multiplier *= dimensions[i]

        if at_center[0] and offset_coord[i] != footprint_center_offset[i]:
            at_center[0] = False

    # Each dimension is in bounds.
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef image_dtype get_neighborhood_peak(
    image_dtype* image_ptr,
    Py_ssize_t* image_dimensions_ptr,
    Py_ssize_t num_dimensions,
    Py_ssize_t* center_coord_ptr,
    uint8_t* footprint_ptr,
    Py_ssize_t* footprint_dimensions_ptr,
    Py_ssize_t* footprint_center_ptr,
    image_dtype border_value,
    uint8_t method,
    Py_ssize_t* footprint_coord_ptr,
    Py_ssize_t* neighbor_coord_ptr,
) nogil:
    """Get the neighborhood peak around a point.

    For dilation, this is the maximum in the neighborhood. For erosion, the minimum.

    The neighborhood is defined by a footprint. Points are excluded if
    the footprint is 0, and included otherwise. The footprint must have
    an odd number of rows and columns, and is anchored at the center.

    border_value is used for out-of-bounds points. In expected usage,
    this is the minimum image value.

    Args:
      image_ptr (image_dtype*): the image to scan
      image_dimensions_ptr (Py_ssize_t*): the size of each dimension
      num_dimensions (Py_ssize_t): the number of image dimensions
      center_coord_ptr (Py_ssize_t*): the coordinates of the point to scan
      footprint_ptr (uint8_t*): the neighborhood footprint
      footprint_dimensions_ptr (Py_ssize_t*): the size of each dimension of the footprint
      footprint_center_ptr (uint8_t*): the offset of the footprint center.
      border_value (image_dtype): the value to use for out-of-bound points
      method (uint8_t): METHOD_DILATION or METHOD_EROSION
      footprint_coord_ptr (Py_ssize_t*): a scratch space for indices
      neighbor_coord_ptr (Py_ssize_t*): a scratch space for neighbor coordinates

    Returns:
        image_dtype: the maximum in the point's neighborhood, greater than or equal to border_value.
    """
    cdef image_dtype pixel_value
    # The peak starts at the border value, and is updated up (or down) as necessary.
    cdef image_dtype neighborhood_peak = border_value
    cdef Py_ssize_t linear_index = 0
    cdef Py_ssize_t neighbor_linear_index = 0

    cdef Py_ssize_t dimension
    cdef uint8_t oob
    cdef uint8_t at_center

    # Reset the loop points to zero.
    for dimension in range(num_dimensions):
        footprint_coord_ptr[dimension] = 0

    while True:
        oob = not offset_coord(
            center_coord_ptr,
            footprint_center_ptr,
            footprint_coord_ptr,
            neighbor_coord_ptr,
            1,
            image_dimensions_ptr,
            num_dimensions,
            &neighbor_linear_index,
            &at_center,
        )

        # Consider in-bound points in the footprint or at the current point.
        if not oob and (at_center or footprint_ptr[linear_index]):
            pixel_value = image_ptr[neighbor_linear_index]
            if method == METHOD_DILATION:
                neighborhood_peak = max(neighborhood_peak, pixel_value)
            elif method == METHOD_EROSION:
                neighborhood_peak = min(neighborhood_peak, pixel_value)

        linear_index += 1
        if increment_index(footprint_coord_ptr, footprint_dimensions_ptr, num_dimensions):
            break

    return neighborhood_peak


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint8_t should_propagate(
    image_dtype* image_ptr,
    Py_ssize_t* image_dimensions_ptr,
    Py_ssize_t num_dimensions,
    image_dtype* mask_ptr,
    Py_ssize_t* coord_ptr,
    image_dtype point_value,
    uint8_t* footprint_ptr,
    Py_ssize_t* footprint_dimensions_ptr,
    Py_ssize_t* footprint_center_ptr,
    uint8_t method,
    Py_ssize_t* footprint_coord_ptr,
    Py_ssize_t* neighbor_coord_ptr,
) nogil:
    """Determine if a point should be propagated to its neighbors.

    This implements the queue test during the raster scan/anti-scan.
    If this function is true, the point's value may need to propagate
    through the image.

    The image and mask must be of the same type and shape. The footprint
    is anchored at the offset point. In the fast-hybrid-reconstruct
    algorithm, the footprint is the raster footprint without the center point.

    Args:
        image_ptr (image_dtype*): the image to scan
        image_dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of image dimensions
        mask_ptr (image_dtype*): the mask to apply
        coord_ptr (Py_ssize_t*): the coordinates of the point to scan
        point_value (image_dtype): the value of the point to scan
        footprint_ptr (uint8_t*): the neighborhood footprint
        footprint_dimensions_ptr (Py_ssize_t*): the size of each dimension of the footprint
        footprint_center_ptr (uint8_t*): the offset of the footprint center.
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        footprint_coord_ptr (Py_ssize_t*): a scratch space for indices
        neighbor_coord_ptr (Py_ssize_t*): a scratch space for neighbor coordinates

    Returns:
        uint8_t: 1 if the point should be propagated, 0 otherwise.
    """
    cdef image_dtype neighbor_value

    cdef Py_ssize_t footprint_linear_index = 0
    cdef Py_ssize_t neighbor_linear_index = 0
    cdef Py_ssize_t dimension
    cdef uint8_t oob
    cdef uint8_t at_center
    cdef Py_ssize_t center_linear_index = point_to_linear(footprint_center_ptr, footprint_dimensions_ptr, num_dimensions)

    # Reset the loop points to zero.
    for dimension in range(num_dimensions):
        footprint_coord_ptr[dimension] = 0

    # For each point the queue point could propagate to, in
    # other words for each point this point is a neighbor of,
    # propagate if necessary & add that point to the queue
    # for further propagation.
    while True:
        # There's no need to test in reverse-raster order during the
        # reverse raster scan: we're already scanning in that order.
        if footprint_linear_index > center_linear_index:
            break

        oob = not offset_coord(
            coord_ptr,
            footprint_center_ptr,
            footprint_coord_ptr,
            neighbor_coord_ptr,
            -1, # we are testing for points of which *this point* is a neighbor
            image_dimensions_ptr,
            num_dimensions,
            &neighbor_linear_index,
            &at_center,
        )

        # Skip:
        # - out-of-bounds points
        # - the center point
        # - points not in the footprint
        if not oob and not at_center and footprint_ptr[footprint_linear_index]:
            neighbor_value = image_ptr[neighbor_linear_index]
            if method == METHOD_DILATION and (
                neighbor_value < point_value
                and neighbor_value < mask_ptr[neighbor_linear_index]
            ):
                return 1
            elif method == METHOD_EROSION and (
                    neighbor_value > point_value
                    and neighbor_value > mask_ptr[neighbor_linear_index]
            ):
                return 1

        footprint_linear_index += 1
        if increment_index(footprint_coord_ptr, footprint_dimensions_ptr, num_dimensions):
            break

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform_raster_scan(
    image_dtype* image_ptr,
    Py_ssize_t* image_dimensions_ptr,
    Py_ssize_t num_dimensions,
    image_dtype* mask_ptr,
    uint8_t* footprint_ptr,
    Py_ssize_t* footprint_dimensions_ptr,
    Py_ssize_t* footprint_center_ptr,
    image_dtype border_value,
    uint8_t method,
):
    cdef image_dtype neighborhood_peak, point_mask
    cdef Py_ssize_t scan_linear_index

    scan_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* scan_coord_ptr = <Py_ssize_t*> <Py_ssize_t> scan_coord_numpy.ctypes.data

    # We use these 2 buffers to avoid re-allocating them in the inner loop.
    loop_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    while True:
        scan_linear_index = point_to_linear(scan_coord_ptr, image_dimensions_ptr, num_dimensions)
        point_mask = <image_dtype> mask_ptr[scan_linear_index]

        # If the image is already at the limiting mask value, skip this pixel.
        if image_ptr[scan_linear_index] == point_mask:
            if increment_index(scan_coord_ptr, image_dimensions_ptr, num_dimensions):
                break
            continue

        neighborhood_peak = get_neighborhood_peak(
            image_ptr,
            image_dimensions_ptr,
            num_dimensions,
            scan_coord_ptr,
            footprint_ptr,
            footprint_dimensions_ptr,
            footprint_center_ptr,
            border_value,
            method,
            loop_coord_ptr,
            neighbor_coord_ptr,
        )

        if method == METHOD_DILATION:
            image_ptr[scan_linear_index] = min(neighborhood_peak, point_mask)
        elif method == METHOD_EROSION:
            image_ptr[scan_linear_index] = max(neighborhood_peak, point_mask)

        if increment_index(scan_coord_ptr, image_dimensions_ptr, num_dimensions):
            break


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform_reverse_raster_scan(
    image_dtype* image_ptr,
    Py_ssize_t* image_dimensions_ptr,
    Py_ssize_t num_dimensions,
    image_dtype* mask_ptr,
    uint8_t* footprint_ptr,
    uint8_t* propagation_footprint_ptr,
    Py_ssize_t* footprint_dimensions_ptr,
    Py_ssize_t* footprint_center_ptr,
    image_dtype border_value,
    uint8_t method,
    queue,
):
    cdef image_dtype neighborhood_peak, point_mask
    cdef Py_ssize_t scan_linear_index
    cdef Py_ssize_t dimension

    scan_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* scan_coord_ptr = <Py_ssize_t*> <Py_ssize_t> scan_coord_numpy.ctypes.data
    for dimension in range(num_dimensions):
        scan_coord_ptr[dimension] = image_dimensions_ptr[dimension] - 1

    # We use these 2 buffers to avoid re-allocating them in the inner loop.
    loop_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    while True:
        scan_linear_index = point_to_linear(scan_coord_ptr, image_dimensions_ptr, num_dimensions)
        point_mask = <image_dtype> mask_ptr[scan_linear_index]

        # If we're already at the mask, skip the neighbor test.
        # But note: we still need to test for propagation (below).
        if image_ptr[scan_linear_index] != point_mask:
            neighborhood_peak = get_neighborhood_peak(
                image_ptr,
                image_dimensions_ptr,
                num_dimensions,
                scan_coord_ptr,
                footprint_ptr,
                footprint_dimensions_ptr,
                footprint_center_ptr,
                border_value,
                method,
                loop_coord_ptr,
                neighbor_coord_ptr,
            )
            if method == METHOD_DILATION:
                image_ptr[scan_linear_index] = min(neighborhood_peak, point_mask)
            elif method == METHOD_EROSION:
                image_ptr[scan_linear_index] = max(neighborhood_peak, point_mask)

        if should_propagate(
                image_ptr,
                image_dimensions_ptr,
                num_dimensions,
                mask_ptr,
                scan_coord_ptr,
                image_ptr[scan_linear_index],
                propagation_footprint_ptr,
                footprint_dimensions_ptr,
                footprint_center_ptr,
                method,
                loop_coord_ptr,
                neighbor_coord_ptr,
        ):
            queue.append(scan_linear_index)

        if decrement_index(scan_coord_ptr, image_dimensions_ptr, num_dimensions):
            break


@cython.boundscheck(False)
@cython.wraparound(False)
cdef process_queue(
    image_dtype* image_ptr,
    Py_ssize_t* image_dimensions_ptr,
    Py_ssize_t num_dimensions,
    image_dtype* mask_ptr,
    uint8_t* footprint_ptr,
    Py_ssize_t* footprint_dimensions_ptr,
    Py_ssize_t* offset_ptr,
    queue,
    uint8_t method,
):
    """Process the queue of pixels to propagate through a image.

    This implements the queue phase of the fast-hybrid reconstruction
    algorithm. During the raster scan phases, we identify pixels that
    may need to propagate through the image. This phase processes
    those queues, propagating the points further as necessary.

    Note that this modifies the image in place.

    Args:
        image_ptr (image_type[][]): the image to scan
        image_dimensions_ptr (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of image dimensions
        mask_ptr (image_dtype*): the image mask (ceiling on image values)
        footprint_ptr (uint8_t*): the neighborhood footprint
        footprint_dimensions_ptr (Py_ssize_t*): the size of each dimension of the footprint
        offset_ptr (uint8_t*): the offset of the footprint center.
        queue (deque): the queue of points to process
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
    """
    cdef Py_ssize_t queue_pt_linear_index
    cdef image_dtype neighbor_mask
    cdef image_dtype neighbor_value, point_value

    coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* queue_pt_coord_ptr = <Py_ssize_t*> <Py_ssize_t> coord_numpy.ctypes.data

    # Pre-allocate these 2 buffers & re-use them in loops later.
    loop_coord_numpy = np.array([0] * num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.array([0] * num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    cdef Py_ssize_t neighbor_linear_index
    cdef Py_ssize_t footprint_linear_index = 0

    cdef uint8_t oob
    cdef uint8_t at_center

    # Process the queue of pixels that need to be updated.
    while len(queue) > 0:
        queue_pt_linear_index = queue.popleft()
        linear_to_point(queue_pt_linear_index, queue_pt_coord_ptr, image_dimensions_ptr, num_dimensions)
        point_value = image_ptr[queue_pt_linear_index]
        footprint_linear_index = 0

        # For each point the queue point could propagate to, in
        # other words for each point this point is a neighbor of,
        # propagate if necessary & add that point to the queue
        # for further propagation.
        while True:
            oob = not offset_coord(
                queue_pt_coord_ptr,
                offset_ptr,
                loop_coord_ptr,
                neighbor_coord_ptr,
                -1, # we are testing for points of which *this point* is a neighbor
                image_dimensions_ptr,
                num_dimensions,
                &neighbor_linear_index,
                &at_center,
            )

            # Skip out of bounds
            # The center point is always skipped.
            # Also skip if not in footprint.
            if not oob and not at_center and footprint_ptr[footprint_linear_index]:
                neighbor_value = image_ptr[neighbor_linear_index]
                neighbor_mask = <image_dtype> mask_ptr[neighbor_linear_index]

                if method == METHOD_DILATION and (point_value > neighbor_value != neighbor_mask):
                    image_ptr[neighbor_linear_index] = min(point_value, neighbor_mask)
                    queue.append(neighbor_linear_index)
                elif method == METHOD_EROSION and (point_value < neighbor_value != neighbor_mask):
                    image_ptr[neighbor_linear_index] = max(point_value, neighbor_mask)
                    queue.append(neighbor_linear_index)

            footprint_linear_index += 1
            if increment_index(loop_coord_ptr, footprint_dimensions_ptr, num_dimensions):
                break


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_reconstruct(
    image,
    mask,
    footprint,
    uint8_t method,
    offset
):
    # The dummy value lets the compiler pick the right overload.
    # (The image & mask are Python numpy objects)
    dummy_value = image.dtype.type(0)

    # To support a new type, add it here and to the type alias.
    if image.dtype == np.uint8:
        fast_hybrid_reconstruct_impl(<uint8_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int8:
        fast_hybrid_reconstruct_impl(<int8_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint16:
        fast_hybrid_reconstruct_impl(<uint16_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int16:
        fast_hybrid_reconstruct_impl(<int16_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint32:
        fast_hybrid_reconstruct_impl(<uint32_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int32:
        fast_hybrid_reconstruct_impl(<int32_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint64:
        fast_hybrid_reconstruct_impl(<uint64_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int64:
        fast_hybrid_reconstruct_impl(<int64_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.float32:
        fast_hybrid_reconstruct_impl(<float> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.float64:
        fast_hybrid_reconstruct_impl(<double> dummy_value, image, mask, footprint, method, offset)
    else:
        raise ValueError("Unsupported image dtype: %s" % image.dtype)

    return image

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fast_hybrid_reconstruct_impl(
    image_dtype _dummy_value,
    image,
    mask,
    footprint,
    uint8_t method,
    offset
):
    """Perform grayscale reconstruction using the 'Fast-Hybrid' algorithm.

    Functionally equivalent to scikit-image's grayreconstruct. That
    implementation uses the Downhill Filter algorithm.

    This implements the fast-hybrid grayscale reconstruction algorithm as
    described by Luc Vincent (1993). That paper isn't publicly available
    but the algorithm is well-described in this paper:

    Teodoro, George. (2012). A fast parallel implementation of queue-
    based morphological reconstruction using gpus.

    https://www.researchgate.net/publication/271444531_A_fast_parallel_implementation_of_queue-based_morphological_reconstruction_using_gpus

    Note however that this implementation does not use the GPU
    enhancements described in Teodoro's paper. Instead, it relies on Cython
    to perform the sequential iterations very quickly. One of its main
    advantages over the downhill filter implementation above is that it
    avoids two n-log-n sorts, and accompanying memory allocations. Instead
    most of its work is performed in 2 linear scans.

    Note that this modifies the image in place.

    Args:
        _dummy_value (image_dtype): a dummy value of the image dtype for type matching
        image (numpy array of type: image_dtype): the image
        mask (numpy array of type: image_dtype): the mask image
        footprint (numpy array of type: uint8_t): the neighborhood footprint aka N(G)
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        offset (numpy array of type: Py_ssize_t): the offset of the footprint center.

    Returns:
        numpy array of type image_dtype: the reconstructed image, modified in place
    """
    cdef image_dtype border_value
    cdef Py_ssize_t num_dimensions = image.ndim

    cdef Py_ssize_t* offset_ptr = <Py_ssize_t*> <Py_ssize_t> offset.ctypes.data

    footprint_dimensions = np.array(footprint.shape, dtype=np.int64)
    cdef Py_ssize_t* footprint_dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> footprint_dimensions.ctypes.data

    # The center point, in 1d linear order.
    cdef Py_ssize_t footprint_linear_center = point_to_linear(offset_ptr, footprint_dimensions_ptr, num_dimensions)

    cdef Py_ssize_t num_before = footprint_linear_center
    cdef Py_ssize_t num_after = np.prod(footprint.shape) - footprint_linear_center - <Py_ssize_t> 1

    # N+(G), the pixels *before* & including the center in a raster scan.
    ones_before = np.concatenate(
        [
            np.ones(num_before + 1, dtype=np.uint8),
            np.zeros(num_after, dtype=np.uint8),
        ]
    ).reshape(footprint.shape)
    footprint_raster_before = (footprint * ones_before).astype(np.uint8)
    # N-(G), the pixels *after* & including the center in a raster scan.
    ones_after = np.concatenate(
        [
            np.zeros(num_before, dtype=bool),
            np.ones(num_after + 1, dtype=bool),
        ]
    ).reshape(footprint.shape)
    footprint_raster_after = (footprint * ones_after).astype(np.uint8)

    # Vincent '93 uses N- as the propagation test.
    # In other words, it checks all q ∈ N-(p) : the points after.
    # The idea is, in our anti-raster scan, do we need to propagate
    # "back" in raster direction. Updating p could affect points
    # which have p as a neighbor.
    # For a symmetric footprint, if q ∈ N-(p) then, by symmetry, p ∈ N+(q).
    # So checking N-(p) is equivalent to checking N+(q).
    #
    # However, an asymmetric footprint doesn't have this property.
    # TODO: I think we only need to check N- because we're scanning back to N+ anyhow
    footprint_propagation_test = np.copy(footprint)

    # .item() converts the numpy scalar to a python scalar
    if method == METHOD_DILATION:
        border_value = np.min(image).item()
    elif method == METHOD_EROSION:
        border_value = np.max(image).item()

    # The propagation queue for after the raster scans.
    queue = deque()

    cdef uint8_t* footprint_ptr = <uint8_t*> <Py_ssize_t> footprint.ctypes.data
    cdef uint8_t* footprint_before_ptr = <uint8_t*> <Py_ssize_t> footprint_raster_before.ctypes.data
    cdef uint8_t* footprint_after_ptr = <uint8_t*> <Py_ssize_t> footprint_raster_after.ctypes.data
    cdef uint8_t* footprint_propagation_ptr = <uint8_t*> <Py_ssize_t> footprint_propagation_test.ctypes.data

    image_dimensions = np.array(image.shape, dtype=np.int64)
    cdef Py_ssize_t* image_dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> image_dimensions.ctypes.data
    cdef image_dtype* image_ptr = <image_dtype*> <Py_ssize_t> image.ctypes.data
    cdef image_dtype* mask_ptr = <image_dtype*> <Py_ssize_t> mask.ctypes.data

    t = timeit.default_timer()
    perform_raster_scan(
        image_ptr,
        image_dimensions_ptr,
        num_dimensions,
        mask_ptr,
        footprint_before_ptr,
        footprint_dimensions_ptr,
        offset_ptr,
        border_value,
        method,
    )
    logging.debug("Raster scan time: %s", timeit.default_timer() - t)

    t = timeit.default_timer()
    perform_reverse_raster_scan(
        image_ptr,
        image_dimensions_ptr,
        num_dimensions,
        mask_ptr,
        footprint_after_ptr,
        footprint_propagation_ptr,
        footprint_dimensions_ptr,
        offset_ptr,
        border_value,
        method,
        queue,
    )
    logging.debug("Reverse raster scan time: %s", timeit.default_timer() - t)

    # Propagate points as necessary.
    logging.debug("Queue size: %s" % len(queue))
    t = timeit.default_timer()
    process_queue(
        image_ptr,
        image_dimensions_ptr,
        num_dimensions,
        mask_ptr,
        footprint_ptr,
        footprint_dimensions_ptr,
        offset_ptr,
        queue,
        method,
    )
    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

    # All done. Image was modified in place.
