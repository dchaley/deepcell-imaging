# cython:language_level=3
# cython:infer_types=True

from collections import deque
import logging
import numpy as np
import timeit
from skimage.morphology import disk

cimport cython
from libc.stdint cimport uint8_t

ctypedef fused my_type:
    int
    float
    double
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
cdef my_type get_neighborhood_max(
    my_type[:, ::1] image,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    uint8_t[:, ::1] footprint,
    Py_ssize_t footprint_center_row,
    Py_ssize_t footprint_center_col,
    my_type border_value,
):
    # OOB values get the border value
    cdef my_type pixel_value
    cdef my_type neighborhood_max = border_value
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef Py_ssize_t footprint_x, footprint_y
    cdef Py_ssize_t footprint_row_offset
    cdef Py_ssize_t footprint_col_offset

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
                or neighbor_row >= image.shape[0]
                or neighbor_col < 0
                or neighbor_col >= image.shape[1]
            ):
                continue
            else:
                pixel_value = image[neighbor_row, neighbor_col]
                neighborhood_max = max(neighborhood_max, pixel_value)

    return neighborhood_max


@cython.boundscheck(False)
@cython.wraparound(False)
cdef should_propagate(
    my_type[:, ::1] image,
    my_type[:, ::1] mask,
    Py_ssize_t point_row,
    Py_ssize_t point_col,
    uint8_t[:, ::1] footprint,
    Py_ssize_t footprint_center_row,
    Py_ssize_t footprint_center_col,
):
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row, neighbor_col
    cdef my_type neighbor_value
    cdef Py_ssize_t footprint_x, footprint_y

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
                or neighbor_row >= image.shape[0]
                or neighbor_col < 0
                or neighbor_col >= image.shape[1]
            ):
                continue

            neighbor_value = image[neighbor_row, neighbor_col]
            if (
                neighbor_value < image[point_row, point_col]
                and neighbor_value < mask[neighbor_row, neighbor_col]
            ):
                return True

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_loop(
    my_type[:, ::1] marker,
    my_type[:, ::1] mask,
    uint8_t[:, ::1] footprint_raster_before,
    uint8_t[:, ::1] footprint_raster_after,
    uint8_t[:, ::1] footprint_propagation_test,
    my_type border_value,
    queue,
):
    cdef Py_ssize_t row, col
    cdef Py_ssize_t marker_rows, marker_cols
    cdef Py_ssize_t footprint_rows, footprint_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef my_type neighborhood_max

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
            # If the marker is already at the maximum mask value, skip this pixel.
            if marker[row, col] == mask[row, col]:
                continue

            neighborhood_max = get_neighborhood_max(
                marker,
                row,
                col,
                footprint_raster_before,
                footprint_center_row,
                footprint_center_col,
                border_value,
            )
            marker[row, col] = min(neighborhood_max, mask[row, col])

    # Scan in reverse-raster order.
    for row in range(marker_rows - 1, -1, -1):
        for col in range(marker_cols - 1, -1, -1):
            neighborhood_max = get_neighborhood_max(
                marker,
                row,
                col,
                footprint_raster_after,
                footprint_center_row,
                footprint_center_col,
                border_value,
            )
            marker[row, col] = min(neighborhood_max, mask[row, col])

            if should_propagate(
                    marker,
                    mask,
                    row,
                    col,
                    footprint_propagation_test,
                    footprint_center_row,
                    footprint_center_col,
            ):
                queue.append((row, col))

    logging.debug("Raster scan/anti-scan time: %s", timeit.default_timer() - t)


@cython.boundscheck(False)
@cython.wraparound(False)
def process_queue(
   my_type[:, ::1] marker,
   Py_ssize_t marker_rows,
   Py_ssize_t marker_cols,
   my_type[:, ::1] mask,
   uint8_t[:, ::1] footprint,
   Py_ssize_t footprint_center_row,
   Py_ssize_t footprint_center_col,
   queue,
):
    cdef Py_ssize_t row, col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset

    # Process the queue of pixels that need to be updated.
    logging.debug("Queue size: %s", len(queue))
    t = timeit.default_timer()
    while len(queue) > 0:
        point = queue.popleft()
        row = point[0]
        col = point[1]

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

                if marker[row, col] > marker[neighbor_row, neighbor_col] != mask[neighbor_row, neighbor_col]:
                    neighbor_coord = (
                        row + footprint_row_offset,
                        col + footprint_col_offset,
                    )
                    marker[neighbor_row, neighbor_col] = min(marker[row, col], mask[neighbor_row, neighbor_col])
                    queue.append(neighbor_coord)

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hybrid_reconstruct(
    my_type[:, ::1] marker, my_type[:, ::1] mask, radius = 2
):
    cdef Py_ssize_t row, col
    cdef Py_ssize_t marker_rows, marker_cols
    cdef Py_ssize_t footprint_center_row, footprint_center_col
    cdef Py_ssize_t footprint_row_offset, footprint_col_offset
    cdef Py_ssize_t neighbor_row
    cdef Py_ssize_t neighbor_col
    cdef my_type border_value
    cdef my_type neighborhood_max

    # N(G), the pixels in our neighborhood.
    footprint = disk(radius)

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

    footprint_propagation_test = np.copy(footprint_raster_after)
    footprint_propagation_test[footprint_center_row, footprint_center_col] = False

    # .item() converts the numpy scalar to a python scalar
    border_value = np.min(marker).item()

    marker_rows = marker.shape[0]
    marker_cols = marker.shape[1]

    queue = deque()

    fast_hybrid_loop(
        marker,
        mask,
        footprint_raster_before,
        footprint_raster_after,
        footprint_propagation_test,
        border_value,
        queue,
    )

    process_queue(
        marker,
        marker_rows,
        marker_cols,
        mask,
        footprint,
        footprint_center_row,
        footprint_center_col,
        queue,
    )


    return marker

