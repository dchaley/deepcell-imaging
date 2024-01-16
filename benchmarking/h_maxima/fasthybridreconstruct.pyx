# cython:language_level=3

from collections import deque
import logging
import numpy as np
import timeit

cimport cython
from libc.stdint cimport uint8_t

# This fast-hybrid reconstruction algorithm supports the following data types.
# Adding more should be a simple matter of adding them to this list.
ctypedef fused marker_dtype:
    uint8_t
    int
    float
    double
    long long
ctypedef fused mask_dtype:
    uint8_t
    int
    float
    double
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
cdef marker_dtype get_neighborhood_max(
    marker_dtype[:, ::1] image,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    uint8_t[:, ::1] footprint,
    marker_dtype border_value,
):
    """Get the neighborhood maximum around a point.

    The neighborhood is defined by a footprint. Points are excluded if
    the footprint is 0, and included otherwise. The footprint must have
    an odd number of rows and columns, and is anchored at the center.

    border_value is used for out-of-bounds points. In expected usage,
    this is the minimum image value.

    Args:
      image (mytype[][]): the image to scan
      point_row (Py_ssize_t): the row of the point to scan
      point_col (Py_ssize_t): the column of the point to scan
      footprint (uint8_t[][]): the neighborhood footprint
      border_value (my_type): the value to use for out-of-bound points

    Returns:
        my_type: the maximum in the point's neighborhood, greater than or equal to border_value.
    """
    cdef marker_dtype pixel_value
    # OOB values get the border value
    cdef marker_dtype neighborhood_max = border_value
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef Py_ssize_t footprint_x, footprint_y
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t image_rows = image.shape[0]
    cdef Py_ssize_t image_cols = image.shape[1]

    cdef Py_ssize_t footprint_center_row = footprint.shape[0] // 2
    cdef Py_ssize_t footprint_center_col = footprint.shape[1] // 2

    for footprint_row_offset in range(-footprint_center_row, footprint_center_row + 1):
        for footprint_col_offset in range(
            -footprint_center_col, footprint_center_col + 1
        ):
            # Skip this point if not in the footprint.
            footprint_x = footprint_center_row + footprint_row_offset
            footprint_y = footprint_center_col + footprint_col_offset
            if not footprint[footprint_x, footprint_y]:
                continue

            neighbor_row = point_row + footprint_row_offset
            neighbor_col = point_col + footprint_col_offset

            if (
                neighbor_row < 0
                or neighbor_row >= image_rows
                or neighbor_col < 0
                or neighbor_col >= image_cols
            ):
                continue
            else:
                pixel_value = image[neighbor_row, neighbor_col]
                neighborhood_max = max(neighborhood_max, pixel_value)

    return neighborhood_max


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint8_t should_propagate(
    marker_dtype[:, ::1] image,
    mask_dtype[:, ::1] mask,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    marker_dtype point_value,
    uint8_t[:, ::1] footprint,
):
    """Determine if a point should be propagated to its neighbors.

    This implements the queue test during the raster scan/anti-scan.
    If this function is true, the point's value may need to propagate
    through the image.

    The image and mask must be of the same type and shape. The footprint
    must have an odd number of rows and columns, and is anchored at the
    center point. In the fast-hybrid-reconstruct algorithm, the
    footprint is the raster footprint but excluding the center point.

    Args:
        image (my_type[][]): the image to scan
        mask (my_type[][]): the mask to scan
        point_row (Py_ssize_t): the row of the point to scan
        point_col (Py_ssize_t): the column of the point to scan
        point_value (my_type): the value of the point to scan
        footprint (uint8_t[][]): the neighborhood footprint

    Returns:
        uint8_t: 1 if the point should be propagated, 0 otherwise.
    """
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef marker_dtype neighbor_value
    cdef Py_ssize_t footprint_x, footprint_y
    cdef Py_ssize_t image_rows = image.shape[0]
    cdef Py_ssize_t image_cols = image.shape[1]
    cdef Py_ssize_t footprint_center_row = footprint.shape[0] // 2
    cdef Py_ssize_t footprint_center_col = footprint.shape[1] // 2

    for footprint_row_offset in range(-footprint_center_row, footprint_center_row + 1):
        for footprint_col_offset in range(
            -footprint_center_col, footprint_center_col + 1
        ):
            # Skip this point if not in the footprint.
            footprint_x = footprint_center_row + footprint_row_offset
            footprint_y = footprint_center_col + footprint_col_offset
            if not footprint[footprint_x, footprint_y]:
                continue

            neighbor_row = point_row + footprint_row_offset
            neighbor_col = point_col + footprint_col_offset

            # Skip out of bounds
            if (
                neighbor_row < 0
                or neighbor_row >= image_rows
                or neighbor_col < 0
                or neighbor_col >= image_cols
            ):
                continue

            neighbor_value = image[neighbor_row, neighbor_col]
            if (
                neighbor_value < point_value
                and neighbor_value < mask[neighbor_row, neighbor_col]
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
    marker_dtype border_value,
    queue,
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
        border_value (my_type): the value to use for out-of-bound points
        queue (deque): the queue of points to process
    """
    cdef Py_ssize_t row, col
    cdef Py_ssize_t marker_rows, marker_cols
    cdef Py_ssize_t footprint_rows, footprint_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef marker_dtype neighborhood_max, point_value
    cdef mask_dtype point_mask

    marker_rows = marker.shape[0]
    marker_cols = marker.shape[1]
    footprint_rows = footprint_raster_before.shape[0]
    footprint_center_row = footprint_rows // 2
    footprint_cols = footprint_raster_before.shape[1]
    footprint_center_col = footprint_cols // 2

    # Scan in raster order.
    t = timeit.default_timer()
    for row in range(marker_rows):
        for col in range(marker_cols):
            point_value = marker[row, col]
            point_mask = mask[row, col]

            # If the marker is already at the maximum mask value, skip this pixel.
            if point_value == point_mask:
                continue

            neighborhood_max = get_neighborhood_max(
                marker,
                row,
                col,
                footprint_raster_before,
                border_value,
            )
            marker[row, col] = <marker_dtype> min(neighborhood_max, point_mask)

    logging.debug("Raster scan time: %s", timeit.default_timer() - t)

    # Scan in reverse-raster order.
    t = timeit.default_timer()
    for row in range(marker_rows - 1, -1, -1):
        for col in range(marker_cols - 1, -1, -1):
            # If we're already at the mask, skip the neighbor test.
            # But note: we still need to test for propagation (below).
            if marker[row, col] != mask[row, col]:
                neighborhood_max = get_neighborhood_max(
                    marker,
                    row,
                    col,
                    footprint_raster_after,
                    border_value,
                )
                point_mask = mask[row, col]
                point_value = <marker_dtype> min(neighborhood_max, point_mask)
                marker[row, col] = point_value

            if should_propagate(
                    marker,
                    mask,
                    row,
                    col,
                    marker[row, col],
                    footprint_propagation_test,
            ):
                queue.append((row, col))

    logging.debug("Raster anti-scan time: %s", timeit.default_timer() - t)


@cython.boundscheck(False)
@cython.wraparound(False)
def process_queue(
   marker_dtype[:, ::1] marker,
   mask_dtype[:, ::1] mask,
   uint8_t[:, ::1] footprint,
   queue,
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
        queue (deque): the queue of points to process
    """
    cdef Py_ssize_t marker_rows = marker.shape[0]
    cdef Py_ssize_t marker_cols = marker.shape[1]
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef mask_dtype neighbor_mask
    cdef marker_dtype neighbor_value, point_value
    cdef Py_ssize_t footprint_center_row = footprint.shape[0] // 2
    cdef Py_ssize_t footprint_center_col = footprint.shape[1] // 2

    # Process the queue of pixels that need to be updated.
    logging.debug("Queue size: %s", len(queue))
    t = timeit.default_timer()
    while len(queue) > 0:
        point = queue.popleft()
        row = point[0]
        col = point[1]
        point_value = marker[row, col]

        for footprint_row_offset in range(
                -footprint_center_row, footprint_center_row + 1
        ):
            for footprint_col_offset in range(
                    -footprint_center_col, footprint_center_col + 1
            ):
                if not footprint[
                    footprint_center_row + footprint_row_offset,
                    footprint_center_col + footprint_col_offset,
                ]:
                    continue

                neighbor_row = row + footprint_row_offset
                neighbor_col = col + footprint_col_offset
                if (
                        neighbor_row < 0
                        or neighbor_row >= marker_rows
                        or neighbor_col < 0
                        or neighbor_col >= marker_cols
                ):
                    # Skip out of bounds
                    continue

                neighbor_value = marker[neighbor_row, neighbor_col]
                neighbor_mask = mask[neighbor_row, neighbor_col]

                if point_value > neighbor_value != neighbor_mask:
                    neighbor_coord = (
                        row + footprint_row_offset,
                        col + footprint_col_offset,
                    )
                    marker[neighbor_row, neighbor_col] = <marker_dtype> min(point_value, neighbor_mask)
                    queue.append(neighbor_coord)

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_reconstruct(
    marker_dtype[:, ::1] marker, mask_dtype[:, ::1] mask, uint8_t[:, ::1] footprint
):
    """Perform grayscale reconstruction using the 'Fast-Hybrid' algorithm.

    Functionally equivalent to scikit-image's grayreconstruct run in
    dilation mode. That implementation uses the Downhill Filter algorithm.

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

    TODO: accept a footprint not a radius. https://github.com/dchaley/deepcell-imaging/issues/30

    Args:
        marker (my_type[][]): the marker image
        mask (my_type[][]): the mask image
        footprint (uint8_t[][]): the neighborhood footprint aka N(G)

    Returns:
        my_type[][]: the reconstructed marker image, modified in place
    """
    cdef Py_ssize_t row, col
    cdef Py_ssize_t marker_rows, marker_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row
    cdef Py_ssize_t neighbor_col
    cdef marker_dtype border_value
    cdef marker_dtype neighborhood_max

    footprint_rows = footprint.shape[0]
    footprint_center_row = footprint_rows // 2
    footprint_cols = footprint.shape[1]
    footprint_center_col = footprint_cols // 2

    # N+(G), the pixels *before* in a raster scan.
    # This is all footprint points before the center.
    ones_before = np.concatenate(
        [
            np.ones((footprint_rows * footprint_cols) // 2 + 1, dtype=bool),
            np.zeros((footprint_rows * footprint_cols) // 2, dtype=bool),
        ]
    ).reshape((footprint_rows, footprint_cols))
    footprint_raster_before = (footprint * ones_before).astype(np.uint8)
    # N-(G), the pixels *after* in a raster scan.
    ones_after = np.concatenate(
        [
            np.zeros((footprint_rows * footprint_cols) // 2, dtype=bool),
            np.ones((footprint_rows * footprint_cols) // 2 + 1, dtype=bool),
        ]
    ).reshape((footprint_rows, footprint_cols))
    footprint_raster_after = (footprint * ones_after).astype(np.uint8)

    # The propagation test is N-(G), but without the center point.
    footprint_propagation_test = np.copy(footprint_raster_after)
    footprint_propagation_test[footprint_center_row, footprint_center_col] = False

    # .item() converts the numpy scalar to a python scalar
    border_value = np.min(marker).item()

    marker_rows = marker.shape[0]
    marker_cols = marker.shape[1]

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
        border_value,
        queue,
    )

    # Propagate points as necessary.
    # The center pixel is excluded from the neighborhood test.
    footprint[footprint_center_row, footprint_center_col] = False
    process_queue(
        marker,
        mask,
        footprint,
        queue,
    )

    # All done. Return marker (which was modified in place).
    return marker
