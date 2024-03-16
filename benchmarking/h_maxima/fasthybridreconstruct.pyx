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
cdef Py_ssize_t point_to_linear(
    Py_ssize_t* point,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
):
    """Convert a point in N dimensions to a linear index.

    Args:
        point (Py_ssize_t*): the point to convert
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        Py_ssize_t: the linear index
    """
    cdef Py_ssize_t linear = 0
    cdef Py_ssize_t multiplier = 1
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        linear += point[i] * multiplier
        multiplier *= dimensions[i]
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
    Py_ssize_t* point_output,
    Py_ssize_t* dimensions,
    Py_ssize_t num_dimensions,
):
    """Convert a linear index to a point in N dimensions.

    Args:
        linear (Py_ssize_t): the linear index
        dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of dimensions

    Returns:
        Py_ssize_t*: the point
    """
    cdef Py_ssize_t i
    for i in range(num_dimensions - 1, -1, -1):
        point_output[i] = cython.cmod(linear, dimensions[i])
        linear = cython.cdiv(linear, dimensions[i])

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
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    uint8_t* footprint,
    Py_ssize_t* footprint_dimensions,
    uint8_t* offset,
    image_dtype border_value,
    uint8_t method,
):
    """Get the neighborhood peak around a point.

    For dilation, this is the maximum in the neighborhood. For erosion, the minimum.

    The neighborhood is defined by a footprint. Points are excluded if
    the footprint is 0, and included otherwise. The footprint must have
    an odd number of rows and columns, and is anchored at the center.

    border_value is used for out-of-bounds points. In expected usage,
    this is the minimum image value.

    Args:
      image (image_dtype*): the image to scan
      image_dimensions (Py_ssize_t*): the size of each dimension
      num_dimensions (Py_ssize_t): the number of image dimensions
      point_row (Py_ssize_t): the row of the point to scan
      point_col (Py_ssize_t): the column of the point to scan
      footprint (uint8_t*): the neighborhood footprint
      footprint_dimensions (Py_ssize_t*): the size of each dimension of the footprint
      offset (uint8_t*): the offset of the footprint center.
      border_value (image_dtype): the value to use for out-of-bound points
      method (uint8_t): METHOD_DILATION or METHOD_EROSION

    Returns:
        image_dtype: the maximum in the point's neighborhood, greater than or equal to border_value.
    """
    cdef image_dtype pixel_value
    # OOB values get the border value
    cdef image_dtype neighborhood_peak = border_value
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef Py_ssize_t footprint_x, footprint_y
    cdef Py_ssize_t offset_row, offset_col

    cdef Py_ssize_t image_rows = image_dimensions[0]
    cdef Py_ssize_t image_cols = image_dimensions[1]
    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]

    cdef Py_ssize_t footprint_rows = footprint_dimensions[0]
    cdef Py_ssize_t footprint_cols = footprint_dimensions[1]

    for offset_row in range(footprint_rows):
        for offset_col in range(footprint_cols):
            # Skip this point if not in the footprint, and not the center point.
            # (The center point is always included in the neighborhood.)
            if (not footprint[offset_row * footprint_cols + offset_col]
                    and not (offset_row == footprint_center_row and offset_col == footprint_center_col)):
                continue

            neighbor_row = point_row + offset_row - footprint_center_row
            neighbor_col = point_col + offset_col - footprint_center_col

            if (
                neighbor_row < 0
                or neighbor_row >= image_rows
                or neighbor_col < 0
                or neighbor_col >= image_cols
            ):
                continue
            else:
                pixel_value = image[neighbor_row * image_cols + neighbor_col]
                if method == METHOD_DILATION:
                    neighborhood_peak = max(neighborhood_peak, pixel_value)
                elif method == METHOD_EROSION:
                    neighborhood_peak = min(neighborhood_peak, pixel_value)

    return neighborhood_peak


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint8_t should_propagate(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    image_dtype* mask,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    image_dtype point_value,
    uint8_t* footprint,
    Py_ssize_t* footprint_dimensions,
    uint8_t* offset,
    uint8_t method,
):
    """Determine if a point should be propagated to its neighbors.

    This implements the queue test during the raster scan/anti-scan.
    If this function is true, the point's value may need to propagate
    through the image.

    The image and mask must be of the same type and shape. The footprint
    is anchored at the offset point. In the fast-hybrid-reconstruct
    algorithm, the footprint is the raster footprint without the center point.

    Args:
        image (image_dtype*): the image to scan
        image_dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of image dimensions
        mask (image_dtype*): the mask to apply
        point_row (Py_ssize_t): the row of the point to scan
        point_col (Py_ssize_t): the column of the point to scan
        point_value (image_dtype): the value of the point to scan
        footprint (uint8_t*): the neighborhood footprint
        footprint_dimensions (Py_ssize_t*): the size of each dimension of the footprint
        offset (uint8_t*): the offset of the footprint center.
        method (uint8_t): METHOD_DILATION or METHOD_EROSION

    Returns:
        uint8_t: 1 if the point should be propagated, 0 otherwise.
    """
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef image_dtype neighbor_value
    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]
    cdef Py_ssize_t footprint_row, footprint_col
    cdef Py_ssize_t image_rows = image_dimensions[0]
    cdef Py_ssize_t image_cols = image_dimensions[1]
    cdef Py_ssize_t footprint_rows = footprint_dimensions[0]
    cdef Py_ssize_t footprint_cols = footprint_dimensions[1]

    # Place the current point at each position of the footprint.
    # If that footprint position is true, then, the current point
    # is a neighbor of the footprint center.
    for footprint_row in range(0, footprint_rows):
        for footprint_col in range(0, footprint_cols):
            # The center point is always skipped.
            # Also skip if not in footprint.
            if ((footprint_row == offset[0] and footprint_col == offset[1])
                    or not footprint[footprint_row * footprint_cols + footprint_col]):
                continue

            # The center point is the current point, offset by the footprint center.
            neighbor_row = point_row + (footprint_center_row - footprint_row)
            neighbor_col = point_col + (footprint_center_col - footprint_col)

            # Skip out of bounds
            if (
                neighbor_row < 0
                or neighbor_row >= image_rows
                or neighbor_col < 0
                or neighbor_col >= image_cols
            ):
                continue

            neighbor_value = image[neighbor_row * image_cols + neighbor_col]
            if method == METHOD_DILATION and (
                neighbor_value < point_value
                and neighbor_value < <image_dtype> mask[neighbor_row * image_cols + neighbor_col]
            ):
                return 1
            elif method == METHOD_EROSION and (
                neighbor_value > point_value
                and neighbor_value > <image_dtype> mask[neighbor_row * image_cols + neighbor_col]
            ):
                return 1

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform_raster_scan(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    image_dtype* mask,
    uint8_t* footprint,
    Py_ssize_t* footprint_dimensions,
    uint8_t* offset,
    image_dtype border_value,
    uint8_t method,
):
    cdef image_dtype neighborhood_peak, point_mask
    cdef Py_ssize_t row, col

    cdef Py_ssize_t image_rows = image_dimensions[0]
    cdef Py_ssize_t image_cols = image_dimensions[1]

    for row in range(image_rows):
        for col in range(image_cols):
            point_mask = <image_dtype> mask[row * image_cols + col]

            # If the image is already at the limiting mask value, skip this pixel.
            if image[row * image_cols + col] == point_mask:
                continue

            neighborhood_peak = get_neighborhood_peak(
                image,
                image_dimensions,
                num_dimensions,
                row,
                col,
                footprint,
                footprint_dimensions,
                offset,
                border_value,
                method,
            )

            if method == METHOD_DILATION:
                image[row * image_cols + col] = min(neighborhood_peak, point_mask)
            elif method == METHOD_EROSION:
                image[row * image_cols + col] = max(neighborhood_peak, point_mask)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void perform_reverse_raster_scan(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    image_dtype* mask,
    uint8_t* footprint,
    uint8_t* propagation_footprint,
    Py_ssize_t* footprint_dimensions,
    uint8_t* offset,
    image_dtype border_value,
    uint8_t method,
    queue,
):
    cdef image_dtype neighborhood_peak, point_mask
    cdef Py_ssize_t row, col

    cdef Py_ssize_t image_rows = image_dimensions[0]
    cdef Py_ssize_t image_cols = image_dimensions[1]

    for row in range(image_rows - 1, -1, -1):
        for col in range(image_cols - 1, -1, -1):
            point_mask = <image_dtype> mask[row * image_cols + col]

            # If we're already at the mask, skip the neighbor test.
            # But note: we still need to test for propagation (below).
            if image[row * image_cols + col] != point_mask:
                neighborhood_peak = get_neighborhood_peak(
                    image,
                    image_dimensions,
                    num_dimensions,
                    row,
                    col,
                    footprint,
                    footprint_dimensions,
                    offset,
                    border_value,
                    method,
                )
                if method == METHOD_DILATION:
                    image[row * image_cols + col] = min(neighborhood_peak, point_mask)
                elif method == METHOD_EROSION:
                    image[row * image_cols + col] = max(neighborhood_peak, point_mask)

            if should_propagate(
                    image,
                    image_dimensions,
                    num_dimensions,
                    mask,
                    row,
                    col,
                    image[row * image_cols + col],
                    propagation_footprint,
                    footprint_dimensions,
                    offset,
                    method,
            ):
                queue.append(row * image_cols + col)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef process_queue(
    image_dtype* image,
    Py_ssize_t* image_dimensions,
    Py_ssize_t num_dimensions,
    image_dtype* mask,
    uint8_t* footprint,
    Py_ssize_t* footprint_dimensions,
    uint8_t* offset,
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
        image (image_type[][]): the image to scan
        image_dimensions (Py_ssize_t*): the size of each dimension
        num_dimensions (Py_ssize_t): the number of image dimensions
        mask (image_dtype*): the image mask (ceiling on image values)
        footprint (uint8_t*): the neighborhood footprint
        footprint_dimensions (Py_ssize_t*): the size of each dimension of the footprint
        offset (uint8_t*): the offset of the footprint center.
        queue (deque): the queue of points to process
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
    """
    cdef Py_ssize_t point_linear
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_row, footprint_col
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef image_dtype neighbor_mask
    cdef image_dtype neighbor_value, point_value
    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]
    cdef Py_ssize_t image_rows = image_dimensions[0]
    cdef Py_ssize_t image_cols = image_dimensions[1]
    cdef Py_ssize_t footprint_rows = footprint_dimensions[0]
    cdef Py_ssize_t footprint_cols = footprint_dimensions[1]

    # Process the queue of pixels that need to be updated.
    logging.debug("Queue size: %s", len(queue))
    t = timeit.default_timer()
    while len(queue) > 0:
        point_linear = queue.popleft()
        row = cython.cdiv(point_linear, image_cols)
        col = cython.cmod(point_linear, image_cols)
        point_value = image[row * image_cols + col]

        # Place the current point at each position of the footprint.
        # If that footprint position is true, then, the current point
        # is a neighbor of the footprint center.
        for footprint_row in range(0, footprint_rows):
            for footprint_col in range(0, footprint_cols):
                # The center point is always skipped.
                # Also skip if not in footprint.
                if ((footprint_row == offset[0] and footprint_col == offset[1])
                        or not footprint[footprint_row * footprint_cols + footprint_col]):
                    continue

                # The center point is the current point, offset by the footprint center.
                neighbor_row = row + (footprint_center_row - footprint_row)
                neighbor_col = col + (footprint_center_col - footprint_col)

                if (
                        neighbor_row < 0
                        or neighbor_row >= image_rows
                        or neighbor_col < 0
                        or neighbor_col >= image_cols
                ):
                    # Skip out of bounds
                    continue

                neighbor_value = image[neighbor_row * image_cols + neighbor_col]
                neighbor_mask = <image_dtype> mask[neighbor_row * image_cols + neighbor_col]

                if method == METHOD_DILATION and (point_value > neighbor_value != neighbor_mask):
                    image[neighbor_row * image_cols + neighbor_col] = min(point_value, neighbor_mask)
                    queue.append(neighbor_row * image_cols + neighbor_col)
                elif method == METHOD_EROSION and (point_value < neighbor_value != neighbor_mask):
                    image[neighbor_row * image_cols + neighbor_col] = max(point_value, neighbor_mask)
                    queue.append(neighbor_row * image_cols + neighbor_col)

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_reconstruct(
    image_dtype[:, ::1] image,
        image_dtype[:, ::1] mask,
        uint8_t[:, ::1] footprint,
        uint8_t method,
        # FIXME(171): offset should be a Py_ssize_t
        uint8_t[::1] offset
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
        image (image_dtype[][]): the image
        mask (image_dtype[][]): the mask image
        footprint (uint8_t[][]): the neighborhood footprint aka N(G)
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        offset (uint8_t[]): the offset of the footprint center.

    Returns:
        image_dtype[][]: the reconstructed image, modified in place
    """
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_rows, footprint_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row
    cdef Py_ssize_t neighbor_col
    cdef image_dtype border_value
    cdef image_dtype neighborhood_peak

    footprint_rows = footprint.shape[0]
    footprint_center_row = offset[0]
    footprint_cols = footprint.shape[1]
    footprint_center_col = offset[1]
    # The center point, in 1d linear order.
    cdef Py_ssize_t linear_center = footprint_center_row * footprint_cols + footprint_center_col

    cdef Py_ssize_t num_before = linear_center
    cdef Py_ssize_t num_after = footprint_rows * footprint_cols - linear_center - 1
    # N+(G), the pixels *before* & including the center in a raster scan.
    ones_before = np.concatenate(
        [
            np.ones(num_before + 1, dtype=np.uint8),
            np.zeros(num_after, dtype=np.uint8),
        ]
    ).reshape((footprint_rows, footprint_cols))
    footprint_raster_before = (footprint * ones_before).astype(np.uint8)
    # N-(G), the pixels *after* & including the center in a raster scan.
    ones_after = np.concatenate(
        [
            np.zeros(num_before, dtype=bool),
            np.ones(num_after + 1, dtype=bool),
        ]
    ).reshape((footprint_rows, footprint_cols))
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

    cdef uint8_t* offset_ptr = &offset[0]
    cdef uint8_t* footprint_before_ptr = <uint8_t*> <Py_ssize_t> footprint_raster_before.ctypes.data
    cdef uint8_t* footprint_after_ptr = <uint8_t*> <Py_ssize_t> footprint_raster_after.ctypes.data
    cdef uint8_t* footprint_propagation_ptr = <uint8_t*> <Py_ssize_t> footprint_propagation_test.ctypes.data

    cdef Py_ssize_t image_rows = image.shape[0]
    cdef Py_ssize_t image_cols = image.shape[1]

    image_dimensions = np.array(image.shape, dtype=np.uint64)
    cdef Py_ssize_t* image_dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> image_dimensions.ctypes.data
    cdef Py_ssize_t num_dimensions = image.ndim

    cdef Py_ssize_t* footprint_dimensions_ptr = <Py_ssize_t*> <Py_ssize_t> footprint.shape

    t = timeit.default_timer()
    perform_raster_scan(
        &image[0, 0],
        image_dimensions_ptr,
        num_dimensions,
        &mask[0, 0],
        footprint_before_ptr,
        footprint_dimensions_ptr,
        offset_ptr,
        border_value,
        method,
    )
    logging.debug("Raster scan time: %s", timeit.default_timer() - t)

    t = timeit.default_timer()
    perform_reverse_raster_scan(
        &image[0, 0],
        image_dimensions_ptr,
        num_dimensions,
        &mask[0, 0],
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
    process_queue(
        &image[0, 0],
        image_dimensions_ptr,
        num_dimensions,
        &mask[0, 0],
        &footprint[0, 0],
        footprint_dimensions_ptr,
        offset_ptr,
        queue,
        method,
    )

    # All done. Return image (which was modified in place).
    return image
