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
    Py_ssize_t* indices,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
) nogil:
    """Increment an index in N dimensions.

    Args:
        indices (Py_ssize_t*): the indices to increment
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        1 if the iteration is complete, 0 otherwise.
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        if indices[i] < dimensions[i] - 1:
            indices[i] += 1
            return 0
        else:
            indices[i] = 0
            if i == 0:
                return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint8_t decrement_index(
        Py_ssize_t* indices,
        Py_ssize_t* dimensions,
        Py_ssize_t num_dimensions,
) nogil:
    """Decrement an index in N dimensions.

    Args:
        indices (Py_ssize_t*): the indices to increment
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        1 if the iteration is complete, 0 otherwise.
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        if indices[i] > 0:
            indices[i] -= 1
            return 0
        else:
            indices[i] = dimensions[i] - 1
            if i == 0:
                return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t coord_to_index(
    Py_ssize_t* coord,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
) nogil:
    """Convert an N-dimensional coordinate to a linear index.

    Args:
        coord (Py_ssize_t*): the coordinate to convert
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        Py_ssize_t: the coordinates as an index
    """
    cdef Py_ssize_t linear = 0
    cdef Py_ssize_t multiplier = 1
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        linear += coord[i] * multiplier
        multiplier *= dimensions[i]
    return linear

# for testing; test_fast_hybrid.py
def point_to_linear_python(point, dimensions, num_dimensions):
    cdef Py_ssize_t* point_ptr = <Py_ssize_t*> <Py_ssize_t> point.ctypes.data
    cdef Py_ssize_t* dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> dimensions.ctypes.data
    cdef Py_ssize_t num_dims = <Py_ssize_t> num_dimensions
    return <long> coord_to_index(point_ptr, dimensions_ptr, num_dims)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void index_to_coord(
    Py_ssize_t index,
    Py_ssize_t* coord_output,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
) nogil:
    """Convert an index to coordinates in N dimensions.

    Args:
        index (Py_ssize_t): the index
        coord_output (Py_ssize_t*): the coordinate buffer to write to
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        coord_output[i] = cython.cmod(index, dimensions[i])
        index = cython.cdiv(index, dimensions[i])

# for testing; test_fast_hybrid.py
def linear_to_point_python(linear, dimensions, num_dimensions):
    point = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* point_output_ptr = <Py_ssize_t*> <Py_ssize_t> point.ctypes.data
    cdef Py_ssize_t* dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> dimensions.ctypes.data
    cdef Py_ssize_t num_dims = <Py_ssize_t> num_dimensions
    index_to_coord(linear, point_output_ptr, dimensions_ptr, num_dims)
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
    Py_ssize_t* result_index,
    uint8_t* at_center,
) nogil:
    """Offset (add) the starting coord by the offset coordinate minus a center offset.

    If the sign is negative, the offset center is subtracted.

    If any dimension is out of bounds, return false and skip assigning more coordinates.

    Otherwise, set all dimension coordinates, the linear index, and whether the coordinate is at the center.

    Returns:
        uint8_t: 1 if the neighbor is in bounds, 0 otherwise.
    """
    cdef Py_ssize_t i
    result_index[0] = 0
    cdef Py_ssize_t multiplier = 1
    at_center[0] = True
    for i in range(num_dimensions - 1, -1, -1):
        result_coord[i] = start_coord[i] + sign * (offset_coord[i] - footprint_center_offset[i])

        # Make sure the dimension is in bounds; return false if not.
        if result_coord[i] < 0 or result_coord[i] >= dimensions[i]:
            return 0

        result_index[0] += result_coord[i] * multiplier
        multiplier *= dimensions[i]

        if at_center[0] and offset_coord[i] != footprint_center_offset[i]:
            at_center[0] = False

    # True: each dimension is in bounds.
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline image_dtype get_neighborhood_peak(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    Py_ssize_t* center_coord,
    uint8_t* footprint_ptr,
    Py_ssize_t footprint_start_index,
    Py_ssize_t footprint_end_index,
    Py_ssize_t* footprint_dimensions,
    Py_ssize_t* footprint_center_coord,
    image_dtype border_value,
    uint8_t method,
    Py_ssize_t* footprint_scan_coord,
    Py_ssize_t* neighbor_coord,
) nogil:
    """Get the neighborhood peak around a coordinate.

    For dilation, this is the maximum in the neighborhood. For erosion, the minimum.

    The neighborhood is defined by a footprint. Points are excluded if
    the footprint is 0, and included otherwise. The footprint must have
    an odd number of rows and columns, and is anchored at the center.

    border_value is used for out-of-bound points.

    Args:
      image (image_dtype*): the image to scan
      image_dimensions (Py_ssize_t*): the size of each dimension
      num_dimensions (Py_ssize_t): the number of image dimensions
      center_coord (Py_ssize_t*): the coordinates of the neighborhood center point
      footprint_ptr (uint8_t*): the neighborhood footprint
      footprint_start_index (Py_ssize_t): the start index of the footprint scan (inclusive)
      footprint_end_index (Py_ssize_t): the end index of the footprint scan (inclusive)
      footprint_dimensions (Py_ssize_t*): the size of each dimension of the footprint
      footprint_center_coord (uint8_t*): the offset of the footprint center.
      border_value (image_dtype): the value to use for out-of-bound points
      method (uint8_t): METHOD_DILATION or METHOD_EROSION
      footprint_scan_coord (Py_ssize_t*): a scratch space for indices
      neighbor_coord (Py_ssize_t*): a scratch space for neighbor coordinates

    Returns:
        image_dtype: the maximum in the point's neighborhood, greater than or equal to border_value.
    """
    cdef image_dtype pixel_value
    # The peak starts at the border value, and is updated up (or down) as necessary.
    cdef image_dtype neighborhood_peak = border_value
    cdef Py_ssize_t neighbor_index = 0

    cdef uint8_t oob
    cdef uint8_t at_center

    # Set the neighborhood loop coordinate to the start.
    cdef Py_ssize_t footprint_scan_index = footprint_start_index
    index_to_coord(footprint_start_index, footprint_scan_coord, footprint_dimensions, num_dimensions)

    while True:
        oob = not offset_coord(
            center_coord,
            footprint_center_coord,
            footprint_scan_coord,
            neighbor_coord,
            1, # we are testing if the point is our neighbor
            image_dimensions,
            num_dimensions,
            &neighbor_index,
            &at_center,
        )

        # Exclude out-of-bound points.
        # Consider neighbors in the footprint, or, the center.
        if not oob and (at_center or footprint_ptr[footprint_scan_index]):
            pixel_value = image[neighbor_index]
            if method == METHOD_DILATION:
                neighborhood_peak = max(neighborhood_peak, pixel_value)
            elif method == METHOD_EROSION:
                neighborhood_peak = min(neighborhood_peak, pixel_value)

        footprint_scan_index += 1
        if footprint_scan_index > footprint_end_index or increment_index(footprint_scan_coord, footprint_dimensions, num_dimensions):
            break

    return neighborhood_peak


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline uint8_t should_propagate(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    image_dtype* mask,
    Py_ssize_t* scan_coord,
    image_dtype scan_value,
    uint8_t* footprint,
    Py_ssize_t* footprint_dimensions,
    Py_ssize_t* footprint_center_coord,
    uint8_t method,
    Py_ssize_t* footprint_coord,
    Py_ssize_t* neighbor_coord,
) nogil:
    """Determine if a point should be propagated to its neighbors.

    This implements the queue test during the raster scan/anti-scan.
    If this function is true, the point's value may need to propagate
    through the image.

    The image and mask must be of the same type and shape. The footprint
    is anchored at the center coord.

    Args:
        image (image_dtype*): the image to scan
        image_dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of image dimensions
        mask (image_dtype*): the mask to apply
        scan_coord (Py_ssize_t*): the coordinates to potentially propagate
        scan_value (image_dtype): the value to potentially propagate
        footprint (uint8_t*): the neighborhood footprint
        footprint_dimensions (Py_ssize_t*): the size of each dimension of the footprint
        footprint_center_coord (uint8_t*): the offset of the footprint center.
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        footprint_coord (Py_ssize_t*): a scratch space for indices
        neighbor_coord (Py_ssize_t*): a scratch space for neighbor coordinates

    Returns:
        uint8_t: 1 if the point should be propagated, 0 otherwise.
    """
    cdef image_dtype neighbor_value

    cdef Py_ssize_t footprint_center_index = coord_to_index(footprint_center_coord, footprint_dimensions, num_dimensions)
    cdef Py_ssize_t footprint_index = 0
    index_to_coord(footprint_index, footprint_coord, footprint_dimensions, num_dimensions)

    cdef Py_ssize_t neighbor_index
    cdef uint8_t oob, at_center

    # For each point the queue point could propagate to, in
    # other words for each point this point is a neighbor of,
    # propagate if necessary & add that point to the queue
    # for further propagation.
    while True:
        oob = not offset_coord(
            scan_coord,
            footprint_center_coord,
            footprint_coord,
            neighbor_coord,
            -1, # we are testing for points of which *this point* is a neighbor
            image_dimensions,
            num_dimensions,
            &neighbor_index,
            &at_center,
        )

        # Skip:
        # - out-of-bounds points
        # - the center point
        # - points not in the footprint
        if not oob and not at_center and footprint[footprint_index]:
            neighbor_value = image[neighbor_index]
            if method == METHOD_DILATION and (
                    neighbor_value < scan_value
                    and neighbor_value < mask[neighbor_index]
            ):
                return 1
            elif method == METHOD_EROSION and (
                    neighbor_value > scan_value
                    and neighbor_value > mask[neighbor_index]
            ):
                return 1

        footprint_index += 1
        # Stop the loop if we've reached the center point.
        # Otherwise, increment and continue.
        if footprint_index > footprint_center_index:
            break
        else:
            increment_index(footprint_coord, footprint_dimensions, num_dimensions)

    return 0


# This function calls the specialized inner function based on the image dtype.
@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_impl(
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
        fast_hybrid_impl_inner(<uint8_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int8:
        fast_hybrid_impl_inner(<int8_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint16:
        fast_hybrid_impl_inner(<uint16_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int16:
        fast_hybrid_impl_inner(<int16_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint32:
        fast_hybrid_impl_inner(<uint32_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int32:
        fast_hybrid_impl_inner(<int32_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.uint64:
        fast_hybrid_impl_inner(<uint64_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.int64:
        fast_hybrid_impl_inner(<int64_t> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.float32:
        fast_hybrid_impl_inner(<float> dummy_value, image, mask, footprint, method, offset)
    elif image.dtype == np.float64:
        fast_hybrid_impl_inner(<double> dummy_value, image, mask, footprint, method, offset)
    else:
        raise ValueError("Unsupported image dtype: %s" % image.dtype)

    return image

# This function takes typed buffers.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fast_hybrid_impl_inner(
    image_dtype _dummy_value,
    image_numpy,
    mask_numpy,
    footprint_numpy,
    uint8_t method,
    footprint_center_numpy
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
        image_numpy (numpy array of type: image_dtype): the image
        mask_numpy (numpy array of type: image_dtype): the mask image
        footprint_numpy (numpy array of type: uint8_t): the neighborhood footprint aka N(G)
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        footprint_center_numpy (numpy array of type: Py_ssize_t): the offset of the footprint center.

    Returns:
        numpy array of type image_dtype: the reconstructed image, modified in place
    """
    cdef image_dtype border_value

    # .item() converts the numpy scalar to a python scalar
    if method == METHOD_DILATION:
        border_value = np.min(image_numpy).item()
    elif method == METHOD_EROSION:
        border_value = np.max(image_numpy).item()
    else:
        raise ValueError("Unknown method: %s" % method)

    # The propagation queue for after the raster scans.
    queue = deque()

    # Get the C buffers from the numpy parameters.
    cdef Py_ssize_t* footprint_center_coord = <Py_ssize_t*> <Py_ssize_t> footprint_center_numpy.ctypes.data
    cdef image_dtype* image = <image_dtype*> <Py_ssize_t> image_numpy.ctypes.data
    cdef image_dtype* mask = <image_dtype*> <Py_ssize_t> mask_numpy.ctypes.data
    cdef uint8_t* footprint = <uint8_t*> <Py_ssize_t> footprint_numpy.ctypes.data

    # Create C buffers to hold the image & footprint dimensions.
    image_dimensions_numpy = np.array(image_numpy.shape, dtype=np.int64)
    footprint_dimensions_numpy = np.array(footprint_numpy.shape, dtype=np.int64)
    cdef Py_ssize_t* image_dimensions = <Py_ssize_t*> <Py_ssize_t> image_dimensions_numpy.ctypes.data
    cdef Py_ssize_t* footprint_dimensions = <Py_ssize_t*> <Py_ssize_t> footprint_dimensions_numpy.ctypes.data

    cdef Py_ssize_t num_dimensions = image_numpy.ndim
    cdef Py_ssize_t footprint_center_index = coord_to_index(footprint_center_coord, footprint_dimensions, num_dimensions)

    # Scan variables.

    # Image & mask values for the current scan or neighbor points.
    cdef image_dtype scan_value, scan_mask, neighbor_value, neighbor_mask
    # The neighborhood peak value.
    cdef image_dtype neighborhood_peak

    # The current scan point coordinates.
    cdef Py_ssize_t scan_index = 0
    scan_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* scan_coord = <Py_ssize_t*> <Py_ssize_t> scan_coord_numpy.ctypes.data

    # The current coordinate in a footprint loop.
    cdef Py_ssize_t footprint_scan_index = 0
    footprint_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* footprint_coord = <Py_ssize_t*> <Py_ssize_t> footprint_coord_numpy.ctypes.data

    # The coordinates of a neighbor point.
    neighbor_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* neighbor_coord = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    ###############
    # Raster scan #
    ###############

    t = timeit.default_timer()
    while True:
        scan_mask = <image_dtype> mask[scan_index]

        # Skip if the image is already at the limiting mask value.
        if image[scan_index] != scan_mask:
            neighborhood_peak = get_neighborhood_peak(
                image,
                image_dimensions,
                num_dimensions,
                scan_coord,
                footprint,
                0,
                footprint_center_index,
                footprint_dimensions,
                footprint_center_coord,
                border_value,
                method,
                footprint_coord,
                neighbor_coord,
            )

            if method == METHOD_DILATION:
                image[scan_index] = min(neighborhood_peak, scan_mask)
            elif method == METHOD_EROSION:
                image[scan_index] = max(neighborhood_peak, scan_mask)

        scan_index += <Py_ssize_t> 1
        if increment_index(scan_coord, image_dimensions, num_dimensions):
            break

    logging.debug("Raster scan time: %s", timeit.default_timer() - t)

    #######################
    # Reverse raster scan #
    #######################

    # Initialize the scan coordinate to the end of the image.
    # Also initialize the footprint end linear index.
    cdef Py_ssize_t dimension
    for dimension in range(num_dimensions - 1, -1, -1):
        scan_coord[dimension] = image_dimensions[dimension] - <Py_ssize_t> 1
        footprint_coord[dimension] = footprint_dimensions[dimension] - <Py_ssize_t> 1

    scan_index = coord_to_index(scan_coord, image_dimensions, num_dimensions)
    footprint_end_index = coord_to_index(footprint_coord, footprint_dimensions, num_dimensions)

    t = timeit.default_timer()
    while True:
        scan_mask = mask[scan_index]

        # If we're already at the mask, skip the neighbor test.
        # But note: we still need to test for propagation (below).
        if image[scan_index] != scan_mask:
            neighborhood_peak = get_neighborhood_peak(
                image,
                image_dimensions,
                num_dimensions,
                scan_coord,
                footprint,
                footprint_center_index,
                footprint_end_index,
                footprint_dimensions,
                footprint_center_coord,
                border_value,
                method,
                footprint_coord,
                neighbor_coord,
            )
            if method == METHOD_DILATION:
                image[scan_index] = min(neighborhood_peak, scan_mask)
            elif method == METHOD_EROSION:
                image[scan_index] = max(neighborhood_peak, scan_mask)

        if should_propagate(
                image,
                image_dimensions,
                num_dimensions,
                mask,
                scan_coord,
                image[scan_index],
                footprint,
                footprint_dimensions,
                footprint_center_coord,
                method,
                footprint_coord,
                neighbor_coord,
        ):
            queue.append(scan_index)

        scan_index -= <Py_ssize_t> 1
        if decrement_index(scan_coord, image_dimensions, num_dimensions):
            break

    logging.debug("Reverse raster scan time: %s", timeit.default_timer() - t)

    #####################
    # Queue propagation #
    #####################

    cdef Py_ssize_t neighbor_index
    cdef uint8_t oob, at_center

    logging.debug("Queue size: %s" % len(queue))
    t = timeit.default_timer()

    while len(queue) > 0:
        scan_index = queue.popleft()
        index_to_coord(scan_index, scan_coord, image_dimensions, num_dimensions)
        footprint_scan_index = 0
        index_to_coord(footprint_scan_index, footprint_coord, footprint_dimensions, num_dimensions)

        # For each point the queue point could propagate to, in
        # other words for each point this point is a neighbor of,
        # propagate if necessary & add that point to the queue
        # for further propagation.
        while True:
            oob = not offset_coord(
                scan_coord,
                footprint_center_coord,
                footprint_coord,
                neighbor_coord,
                -1, # we are testing for points of which *this point* is a neighbor
                image_dimensions,
                num_dimensions,
                &neighbor_index,
                &at_center,
            )

            # Skip:
            # - out-of-bounds points
            # - the center point
            # - points not in the footprint
            if not oob and not at_center and footprint[footprint_scan_index]:
                scan_value = image[scan_index]
                neighbor_value = image[neighbor_index]
                neighbor_mask = mask[neighbor_index]

                if method == METHOD_DILATION and (scan_value > neighbor_value != neighbor_mask):
                    image[neighbor_index] = min(scan_value, neighbor_mask)
                    queue.append(neighbor_index)
                elif method == METHOD_EROSION and (scan_value < neighbor_value != neighbor_mask):
                    image[neighbor_index] = max(scan_value, neighbor_mask)
                    queue.append(neighbor_index)

            footprint_scan_index += <Py_ssize_t> 1
            if increment_index(footprint_coord, footprint_dimensions, num_dimensions):
                break

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

    # All done. Image was modified in place.
