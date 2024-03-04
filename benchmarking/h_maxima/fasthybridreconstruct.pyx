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
ctypedef fused marker_dtype:
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
ctypedef fused mask_dtype:
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
# ctypedef fused marker_dtype:
#     uint8_t
#     int16_t
#     int64_t
#     double
# ctypedef fused mask_dtype:
#     uint8_t
#     int16_t
#     int64_t
#     double

cpdef enum:
    METHOD_DILATION = 0
    METHOD_EROSION = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef marker_dtype get_neighborhood_peak(
    marker_dtype* image,
    Py_ssize_t image_rows,
    Py_ssize_t image_cols,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    uint8_t[:, ::1] footprint,
    uint8_t[::1] offset,
    marker_dtype border_value,
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
      image (marker_dtype*): the image to scan
      image_rows (Py_ssize_t): the number of rows in the image
      image_cols (Py_ssize_t): the number of columns in the image
      point_row (Py_ssize_t): the row of the point to scan
      point_col (Py_ssize_t): the column of the point to scan
      footprint (uint8_t[][]): the neighborhood footprint
      offset (uint8_t[]): the offset of the footprint center.
      border_value (my_type): the value to use for out-of-bound points
      method (uint8_t): METHOD_DILATION or METHOD_EROSION

    Returns:
        my_type: the maximum in the point's neighborhood, greater than or equal to border_value.
    """
    cdef marker_dtype pixel_value
    # OOB values get the border value
    cdef marker_dtype neighborhood_peak = border_value
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef Py_ssize_t footprint_x, footprint_y
    cdef Py_ssize_t offset_row, offset_col

    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]

    for offset_row in range(-footprint_center_row, footprint.shape[0] - footprint_center_row):
        for offset_col in range(-footprint_center_col, footprint.shape[1] - footprint_center_col):
            # Skip this point if not in the footprint, and not the center point.
            # (The center point is always included in the neighborhood.)
            if (not footprint[footprint_center_row + offset_row, footprint_center_col + offset_col]
                    and not (offset_row == 0 and offset_col == 0)):
                continue

            neighbor_row = point_row + offset_row
            neighbor_col = point_col + offset_col

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
    marker_dtype[:, ::1] image,
    mask_dtype[:, ::1] mask,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    marker_dtype point_value,
    uint8_t[:, ::1] footprint,
    uint8_t[::1] offset,
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
        image (marker_dtype[][]): the image to scan
        mask (mask_dtype[][]): the mask to scan
        point_row (Py_ssize_t): the row of the point to scan
        point_col (Py_ssize_t): the column of the point to scan
        point_value (marker_dtype): the value of the point to scan
        footprint (uint8_t[][]): the neighborhood footprint
        offset (uint8_t[]): the offset of the footprint center.
        method (uint8_t): METHOD_DILATION or METHOD_EROSION

    Returns:
        uint8_t: 1 if the point should be propagated, 0 otherwise.
    """
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef marker_dtype neighbor_value
    cdef Py_ssize_t image_rows = image.shape[0]
    cdef Py_ssize_t image_cols = image.shape[1]
    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]
    cdef Py_ssize_t footprint_row, footprint_col

    # Place the current point at each position of the footprint.
    # If that footprint position is true, then, the current point
    # is a neighbor of the footprint center.
    for footprint_row in range(0, footprint.shape[0]):
        for footprint_col in range(0, footprint.shape[1]):
            # The center point is always skipped.
            # Also skip if not in footprint.
            if ((footprint_row == offset[0] and footprint_col == offset[1])
                    or not footprint[footprint_row, footprint_col]):
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

            neighbor_value = image[neighbor_row, neighbor_col]
            if method == METHOD_DILATION and (
                neighbor_value < point_value
                and neighbor_value < <marker_dtype> mask[neighbor_row, neighbor_col]
            ):
                return 1
            elif method == METHOD_EROSION and (
                neighbor_value > point_value
                and neighbor_value > <marker_dtype> mask[neighbor_row, neighbor_col]
            ):
                return 1

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_raster_scans(
    marker_dtype[:, ::1] marker,
    mask_dtype[:, ::1] mask,
    uint8_t[:, ::1] footprint_raster_before,
    uint8_t[:, ::1] footprint_raster_after,
    uint8_t[:, ::1] footprint_propagation_test,
    uint8_t[::1] offset,
    marker_dtype border_value,
    queue,
    uint8_t method,
):
    """Apply a maximum filter in raster order, then in reverse-raster order.

    This implements the raster scan and anti-scan portions of the fast
    hybrid reconstruct algorithm. The scan applies a maximum filter to
    each point, using the pixels before or after the point depending on
    the scan order.

    After being scanned in both orders, each point is tested for further
    propagation. If so, the point is added to the provided queue.

    Note that this modifies the marker image in place.

    Args:
        marker (my_type[][]): the image to scan
        mask (my_type[][]): the mask to scan
        footprint_raster_before (uint8_t[][]): the raster footprint before the center point
        footprint_raster_after (uint8_t[][]): the raster footprint after the center point
        footprint_propagation_test (uint8_t[][]): the raster footprint after the center point, excluding the center point
        offset (uint8_t[]): the offset of the footprint center.
        border_value (my_type): the value to use for out-of-bound points
        queue (deque): the queue of points to process
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
    """
    cdef Py_ssize_t row, col
    cdef Py_ssize_t marker_rows, marker_cols
    cdef marker_dtype neighborhood_peak, point_value, point_mask

    marker_rows = marker.shape[0]
    marker_cols = marker.shape[1]

    # Scan in raster order.
    t = timeit.default_timer()
    for row in range(marker_rows):
        for col in range(marker_cols):
            point_mask = <marker_dtype> mask[row, col]

            # If the marker is already at the limiting mask value, skip this pixel.
            if marker[row, col] == point_mask:
                continue

            neighborhood_peak = get_neighborhood_peak(
                &marker[0, 0],
                marker.shape[0],
                marker.shape[1],
                row,
                col,
                footprint_raster_before,
                offset,
                border_value,
                method,
            )

            if method == METHOD_DILATION:
                marker[row, col] = min(neighborhood_peak, point_mask)
            elif method == METHOD_EROSION:
                marker[row, col] = max(neighborhood_peak, point_mask)

    logging.debug("Raster scan time: %s", timeit.default_timer() - t)

    # Scan in reverse-raster order.
    t = timeit.default_timer()
    for row in range(marker_rows - 1, -1, -1):
        for col in range(marker_cols - 1, -1, -1):
            point_mask = <marker_dtype> mask[row, col]

            # If we're already at the mask, skip the neighbor test.
            # But note: we still need to test for propagation (below).
            if marker[row, col] != point_mask:
                neighborhood_peak = get_neighborhood_peak(
                    &marker[0,0],
                    marker.shape[0],
                    marker.shape[1],
                    row,
                    col,
                    footprint_raster_after,
                    offset,
                    border_value,
                    method,
                )
                if method == METHOD_DILATION:
                    marker[row, col] = min(neighborhood_peak, point_mask)
                elif method == METHOD_EROSION:
                    marker[row, col] = max(neighborhood_peak, point_mask)

            if should_propagate(
                    marker,
                    mask,
                    row,
                    col,
                    marker[row, col],
                    footprint_propagation_test,
                    offset,
                    method,
            ):
                queue.append((row, col))

    logging.debug("Raster anti-scan time: %s", timeit.default_timer() - t)


@cython.boundscheck(False)
@cython.wraparound(False)
def process_queue(
   marker_dtype[:, ::1] marker,
   mask_dtype[:, ::1] mask,
   uint8_t[:, ::1] footprint,
   uint8_t[::1] offset,
   queue,
   uint8_t method,
):
    """Process the queue of pixels to propagate through a marker image.

    This implements the queue phase of the fast-hybrid reconstruction
    algorithm. During the raster scan phases, we identify pixels that
    may need to propagate through the image. This phase processes
    those queues, propagating the points further as necessary.

    Note that this modifies the marker image in place.

    Args:
        marker (mytype[][]): the marker image to scan
        mask (mytype[][]): the image mask (ceiling on image values)
        footprint (uint8_t[][]): the neighborhood footprint
        offset (uint8_t[]): the offset of the footprint center.
        queue (deque): the queue of points to process
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
    """
    cdef Py_ssize_t marker_rows = marker.shape[0]
    cdef Py_ssize_t marker_cols = marker.shape[1]
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_row, footprint_col
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef marker_dtype neighbor_mask
    cdef marker_dtype neighbor_value, point_value
    cdef Py_ssize_t footprint_center_row = offset[0]
    cdef Py_ssize_t footprint_center_col = offset[1]

    # Process the queue of pixels that need to be updated.
    logging.debug("Queue size: %s", len(queue))
    t = timeit.default_timer()
    while len(queue) > 0:
        point = queue.popleft()
        row = point[0]
        col = point[1]
        point_value = marker[row, col]

        # Place the current point at each position of the footprint.
        # If that footprint position is true, then, the current point
        # is a neighbor of the footprint center.
        for footprint_row in range(0, footprint.shape[0]):
            for footprint_col in range(0, footprint.shape[1]):
                # The center point is always skipped.
                # Also skip if not in footprint.
                if ((footprint_row == offset[0] and footprint_col == offset[1])
                        or not footprint[footprint_row, footprint_col]):
                    continue

                # The center point is the current point, offset by the footprint center.
                neighbor_row = row + (footprint_center_row - footprint_row)
                neighbor_col = col + (footprint_center_col - footprint_col)

                if (
                        neighbor_row < 0
                        or neighbor_row >= marker_rows
                        or neighbor_col < 0
                        or neighbor_col >= marker_cols
                ):
                    # Skip out of bounds
                    continue

                neighbor_value = marker[neighbor_row, neighbor_col]
                neighbor_mask = <marker_dtype> mask[neighbor_row, neighbor_col]

                if method == METHOD_DILATION and (point_value > neighbor_value != neighbor_mask):
                    marker[neighbor_row, neighbor_col] = min(point_value, neighbor_mask)
                    queue.append((neighbor_row, neighbor_col))
                elif method == METHOD_EROSION and (point_value < neighbor_value != neighbor_mask):
                    marker[neighbor_row, neighbor_col] = max(point_value, neighbor_mask)
                    queue.append((neighbor_row, neighbor_col))

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_reconstruct(
    marker_dtype[:, ::1] marker,
        mask_dtype[:, ::1] mask,
        uint8_t[:, ::1] footprint,
        uint8_t method,
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

    Note that this modifies the marker image in place.

    Args:
        marker (my_type[][]): the marker image
        mask (my_type[][]): the mask image
        footprint (uint8_t[][]): the neighborhood footprint aka N(G)
        method (uint8_t): METHOD_DILATION or METHOD_EROSION
        offset (uint8_t[]): the offset of the footprint center.

    Returns:
        my_type[][]: the reconstructed marker image, modified in place
    """
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_rows, footprint_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row
    cdef Py_ssize_t neighbor_col
    cdef marker_dtype border_value
    cdef marker_dtype neighborhood_peak

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
        border_value = np.min(marker).item()
    elif method == METHOD_EROSION:
        border_value = np.max(marker).item()

    # The propagation queue for after the raster scans.
    queue = deque()

    # Apply the maximum filter in raster order, then in reverse-raster order.
    # The center pixel is included in both of these tests.
    fast_hybrid_raster_scans(
        marker,
        mask,
        footprint_raster_before,
        footprint_raster_after,
        footprint_propagation_test,
        offset,
        border_value,
        queue,
        method,
    )

    # Propagate points as necessary.
    process_queue(
        marker,
        mask,
        footprint,
        offset,
        queue,
        method,
    )

    # All done. Return marker (which was modified in place).
    return marker
