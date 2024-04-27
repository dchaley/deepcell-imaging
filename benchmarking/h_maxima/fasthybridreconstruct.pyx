# cython:language_level=3

from collections import deque
import logging
import numpy as np
import timeit

cimport cython
from libc.stdint cimport uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t

# This fast-hybrid reconstruction algorithm supports the following data types.
# Adding more should be a simple matter of adding them to this list.

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

# Dev mode types
# ctypedef fused image_dtype:
#     uint8_t
#     int16_t
#     int64_t
#     double

cpdef enum:
    METHOD_DILATION = 0
    METHOD_EROSION = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint8_t increment_index(
    Py_ssize_t* indices_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
):
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
cdef Py_ssize_t point_to_linear(
    Py_ssize_t* coord_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
):
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
cdef Py_ssize_t* linear_to_point(
    Py_ssize_t linear,
    Py_ssize_t* point_output_ptr,
    Py_ssize_t* dimensions_ptr,
    Py_ssize_t num_dimensions,
):
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
):
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
    cdef Py_ssize_t multiplier = 1

    cdef Py_ssize_t dim
    cdef uint8_t end = False
    cdef uint8_t oob
    cdef uint8_t at_center
    cdef uint8_t out_of_footprint
    cdef Py_ssize_t i

    # Reset the loop points to zero.
    for x in range(num_dimensions):
        footprint_coord_ptr[x] = 0

    while True:
        # This gets set to true if necessary.
        oob = False
        # This gets set to false if necessary.
        at_center = True
        # This gets set to true if necessary.
        out_of_footprint = False
        neighbor_linear_index = 0
        multiplier = 1

        for dim in range(num_dimensions - 1, -1, -1):
            # Calculate the neighbor's coordinates
            neighbor_coord_ptr[dim] = center_coord_ptr[dim] + footprint_coord_ptr[dim] - footprint_center_ptr[dim]
            if neighbor_coord_ptr[dim] < 0 or neighbor_coord_ptr[dim] >= image_dimensions_ptr[dim]:
                oob = True
                break
            neighbor_linear_index += neighbor_coord_ptr[dim] * multiplier
            multiplier *= image_dimensions_ptr[dim]
            if at_center and footprint_coord_ptr[dim] != footprint_center_ptr[dim]:
                at_center = False

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
):
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

    cdef Py_ssize_t linear_footprint_index = 0
    cdef Py_ssize_t linear_neighbor_index = 0
    cdef Py_ssize_t multiplier = 1
    cdef Py_ssize_t dim
    cdef uint8_t oob
    cdef uint8_t at_center
    cdef uint8_t out_of_footprint
    cdef uint8_t end = False

    # Reset the loop points to zero.
    for x in range(num_dimensions):
        footprint_coord_ptr[x] = 0

    # Place the current point at each position of the footprint.
    # If that footprint position is true, then, the current point
    # is a neighbor of the footprint center.
    while True:
        oob = False
        at_center = True
        out_of_footprint = False
        linear_neighbor_index = 0
        multiplier = 1

        for dim in range(num_dimensions - 1, -1, -1):
            # The center point is the current point, offset by the footprint center.
            neighbor_coord_ptr[dim] = coord_ptr[dim] + footprint_center_ptr[dim] - footprint_coord_ptr[dim]

            if neighbor_coord_ptr[dim] < 0 or neighbor_coord_ptr[dim] >= image_dimensions_ptr[dim]:
                oob = True
                break

            linear_neighbor_index += neighbor_coord_ptr[dim] * multiplier
            multiplier *= image_dimensions_ptr[dim]

            if at_center and footprint_coord_ptr[dim] != footprint_center_ptr[dim]:
                at_center = False

        # Skip:
        # - out-of-bounds points
        # - the center point
        # - points not in the footprint
        if not oob and not at_center and footprint_ptr[linear_footprint_index]:
            neighbor_value = image_ptr[linear_neighbor_index]
            if method == METHOD_DILATION and (
                neighbor_value < point_value
                and neighbor_value < mask_ptr[linear_neighbor_index]
            ):
                return 1
            elif method == METHOD_EROSION and (
                    neighbor_value > point_value
                    and neighbor_value > mask_ptr[linear_neighbor_index]
            ):
                return 1

        linear_footprint_index += 1
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
    cdef Py_ssize_t linear_index

    scan_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* scan_coord_ptr = <Py_ssize_t*> <Py_ssize_t> scan_coord_numpy.ctypes.data

    # We use these 2 buffers to avoid re-allocating them in the inner loop.
    loop_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    while True:
        linear_index = point_to_linear(scan_coord_ptr, image_dimensions_ptr, num_dimensions)
        point_mask = <image_dtype> mask_ptr[linear_index]

        # If the image is already at the limiting mask value, skip this pixel.
        if image_ptr[linear_index] == point_mask:
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
            image_ptr[linear_index] = min(neighborhood_peak, point_mask)
        elif method == METHOD_EROSION:
            image_ptr[linear_index] = max(neighborhood_peak, point_mask)

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
    cdef Py_ssize_t linear_index

    scan_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* scan_coord_ptr = <Py_ssize_t*> <Py_ssize_t> scan_coord_numpy.ctypes.data

    # We use these 2 buffers to avoid re-allocating them in the inner loop.
    loop_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    while True:
        linear_index = point_to_linear(scan_coord_ptr, image_dimensions_ptr, num_dimensions)
        point_mask = <image_dtype> mask_ptr[linear_index]

        # If we're already at the mask, skip the neighbor test.
        # But note: we still need to test for propagation (below).
        if image_ptr[linear_index] != point_mask:
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
                image_ptr[linear_index] = min(neighborhood_peak, point_mask)
            elif method == METHOD_EROSION:
                image_ptr[linear_index] = max(neighborhood_peak, point_mask)

        if should_propagate(
                image_ptr,
                image_dimensions_ptr,
                num_dimensions,
                mask_ptr,
                scan_coord_ptr,
                image_ptr[linear_index],
                propagation_footprint_ptr,
                footprint_dimensions_ptr,
                footprint_center_ptr,
                method,
                loop_coord_ptr,
                neighbor_coord_ptr,
        ):
            queue.append(linear_index)

        if increment_index(scan_coord_ptr, image_dimensions_ptr, num_dimensions):
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
    cdef Py_ssize_t queue_linear_index
    cdef image_dtype neighbor_mask
    cdef image_dtype neighbor_value, point_value

    coord_numpy = np.zeros(num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* coord_ptr = <Py_ssize_t*> <Py_ssize_t> coord_numpy.ctypes.data

    # We use these 2 buffers to avoid re-allocating them in the inner loop.
    loop_coord_numpy = np.array([0] * num_dimensions, dtype=np.int64)
    neighbor_coord_numpy = np.array([0] * num_dimensions, dtype=np.int64)
    cdef Py_ssize_t* loop_coord_ptr = <Py_ssize_t*> <Py_ssize_t> loop_coord_numpy.ctypes.data
    cdef Py_ssize_t* neighbor_coord_ptr = <Py_ssize_t*> <Py_ssize_t> neighbor_coord_numpy.ctypes.data

    cdef Py_ssize_t neighbor_linear_index
    cdef Py_ssize_t footprint_linear_index

    cdef Py_ssize_t dim
    cdef uint8_t oob
    cdef uint8_t at_center
    cdef uint8_t out_of_footprint

    # Process the queue of pixels that need to be updated.
    while len(queue) > 0:
        queue_linear_index = queue.popleft()
        linear_to_point(queue_linear_index, coord_ptr, image_dimensions_ptr, num_dimensions)
        point_value = image_ptr[queue_linear_index]

        # Place the current point at each position of the footprint.
        # If that footprint position is true, then, the current point
        # is a neighbor of the footprint center.
        while True:
            # This gets set to true if necessary.
            oob = False
            # This gets set to false if necessary.
            at_center = True
            # This gets set to true if necessary.
            out_of_footprint = False

            for dim in range(num_dimensions):
                # Calculate the neighbor's coordinates
                neighbor_coord_ptr[dim] = coord_ptr[dim] + (offset_ptr[dim] - loop_coord_ptr[dim])
                if neighbor_coord_ptr[dim] < 0 or neighbor_coord_ptr[dim] >= image_dimensions_ptr[dim]:
                    oob = True
                    break
                if loop_coord_ptr[dim] != offset_ptr[dim]:
                    at_center = False

            if not oob and not at_center:
                footprint_linear_index = point_to_linear(loop_coord_ptr, footprint_dimensions_ptr, num_dimensions)
                if not footprint_ptr[footprint_linear_index]:
                    out_of_footprint = True

            # Skip out of bounds
            # The center point is always skipped.
            # Also skip if not in footprint.
            if not (oob or at_center or out_of_footprint):
                neighbor_linear_index = point_to_linear(neighbor_coord_ptr, image_dimensions_ptr, num_dimensions)
                neighbor_value = image_ptr[neighbor_linear_index]
                neighbor_mask = <image_dtype> mask_ptr[neighbor_linear_index]

                if method == METHOD_DILATION and (point_value > neighbor_value != neighbor_mask):
                    image_ptr[neighbor_linear_index] = min(point_value, neighbor_mask)
                    queue.append(neighbor_linear_index)
                elif method == METHOD_EROSION and (point_value < neighbor_value != neighbor_mask):
                    image_ptr[neighbor_linear_index] = max(point_value, neighbor_mask)
                    queue.append(neighbor_linear_index)

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
    dummy_value = image.dtype.type(0)

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
    avoids two sorts and accompanying memory allocations for several-
    hundred-megabyte image files.

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
    cdef Py_ssize_t num_after = np.prod(footprint.shape) - footprint_linear_center - 1

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
