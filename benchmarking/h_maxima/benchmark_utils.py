from collections import deque
import cv2
import logging
import numpy as np
from skimage.morphology import h_maxima
from skimage.morphology import disk
import timeit

from fasthybridreconstruct import fast_hybrid_reconstruct


def scikit_h_maxima(image, h=0.075, radius=2):
    return h_maxima(image=image, h=h, footprint=disk(radius))


def opencv_reconstruct(
    marker: np.ndarray, mask: np.ndarray, kernel: np.ndarray, anchor: tuple = (-1, -1)
):
    # .item() converts the numpy scalar to a python scalar
    pad_value = np.min(marker).item()

    if anchor == (-1, -1):
        # OpenCV uses (x, y) not (row, col) for the anchor
        anchor = (kernel.shape[1] // 2, kernel.shape[0] // 2)

    # The kernel however uses rows/cols not x/y…
    if not kernel[anchor[1], anchor[0]]:
        raise ValueError("The center of the kernel must be True")

    while True:
        expanded = cv2.dilate(
            src=marker,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=pad_value,
            anchor=anchor,
        )
        expanded = np.fmin(expanded, mask)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded

        marker = expanded


def get_neighborhood_max(
    image: np.ndarray,
    point_row,
    point_col,
    footprint,
    footprint_center_row,
    footprint_center_col,
    border_value,
):
    # OOB values get the border value
    neighborhood_max = border_value
    for footprint_row_offset in range(-footprint_center_row, footprint_center_row + 1):
        for footprint_col_offset in range(
            -footprint_center_col, footprint_center_col + 1
        ):
            # Skip this point if not in the footprint.
            footprint_x = footprint_center_row + footprint_row_offset
            footprint_y = footprint_center_col + footprint_col_offset
            if not footprint[footprint_x, footprint_y]:
                continue

            neighbor_coord = (
                point_row + footprint_row_offset,
                point_col + footprint_col_offset,
            )

            if (
                neighbor_coord[0] < 0
                or neighbor_coord[0] >= image.shape[0]
                or neighbor_coord[1] < 0
                or neighbor_coord[1] >= image.shape[1]
            ):
                continue
            else:
                pixel_value = image[neighbor_coord]
                neighborhood_max = max(neighborhood_max, pixel_value)

    return neighborhood_max


def should_propagate(
    image: np.ndarray,
    mask: np.ndarray,
    point_row,
    point_col,
    footprint,
    footprint_center_row,
    footprint_center_col,
):
    for footprint_row_offset in range(-footprint_center_row, footprint_center_row + 1):
        for footprint_col_offset in range(
            -footprint_center_col, footprint_center_col + 1
        ):
            # Skip this point if not in the footprint.
            footprint_x = footprint_center_row + footprint_row_offset
            footprint_y = footprint_center_col + footprint_col_offset
            if not footprint[footprint_x, footprint_y]:
                continue

            neighbor_coord = (
                point_row + footprint_row_offset,
                point_col + footprint_col_offset,
            )

            # Skip out of bounds
            if (
                neighbor_coord[0] < 0
                or neighbor_coord[0] >= image.shape[0]
                or neighbor_coord[1] < 0
                or neighbor_coord[1] >= image.shape[1]
            ):
                continue

            neighbor_value = image[neighbor_coord]
            if (
                neighbor_value < image[point_row, point_col]
                and neighbor_value < mask[neighbor_coord]
            ):
                return True

    return False


def reconstruct_fast_hybrid_python(
    marker: np.ndarray, mask: np.ndarray, radius: int = 2
):
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
    footprint_raster_before = footprint * ones_before
    # N-(G), the pixels *after* in a raster scan.
    ones_after = np.concatenate(
        [
            np.zeros((footprint_rows * footprint_cols) // 2, dtype=bool),
            np.ones((footprint_rows * footprint_cols) // 2 + 1, dtype=bool),
        ]
    ).reshape((footprint_rows, footprint_cols))
    footprint_raster_after = footprint * ones_after

    footprint_propagation_test = np.copy(footprint_raster_after)
    footprint_propagation_test[footprint_center_row, footprint_center_col] = False

    # .item() converts the numpy scalar to a python scalar
    border_value = np.min(marker).item()

    marker_rows = marker.shape[0]
    marker_cols = marker.shape[1]

    queue = deque()

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

    # Process the queue of pixels that need to be updated.
    logging.debug("Queue size: %s", len(queue))
    t = timeit.default_timer()
    while len(queue) > 0:
        point = queue.popleft()

        for footprint_row_offset in range(
            -footprint_center_row, footprint_center_row + 1
        ):
            for footprint_col_offset in range(
                -footprint_center_col, footprint_center_col + 1
            ):
                # Don't consider self as neighbor.
                if footprint_row_offset == 0 and footprint_col_offset == 0:
                    continue

                if not footprint_raster_after[
                    footprint_center_row + footprint_row_offset,
                    footprint_center_col + footprint_col_offset,
                ]:
                    continue

                if (
                    row + footprint_row_offset < 0
                    or row + footprint_row_offset >= marker_rows
                    or col + footprint_col_offset < 0
                    or col + footprint_col_offset >= marker_cols
                ):
                    # Skip out of bounds
                    continue

                neighbor_coord = (
                    row + footprint_row_offset,
                    col + footprint_col_offset,
                )
                if marker[point] > marker[neighbor_coord] != mask[neighbor_coord]:
                    marker[neighbor_coord] = min(marker[point], mask[neighbor_coord])
                    queue.append(neighbor_coord)

    logging.debug("Queue processing time: %s", timeit.default_timer() - t)

    return marker


def opencv_h_maxima(image, h=0.075, radius=2):
    # This is mostly copied from scikit h_maxima
    # except using our own grayscale reconstruction
    resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
    shifted_img = image - h - resolution
    reconstructed = opencv_reconstruct(shifted_img, image, radius)
    residue_img = image - reconstructed
    return (residue_img >= h).astype(np.uint8)


def python_h_maxima(image, h=0.075, radius=2):
    # This is mostly copied from scikit h_maxima
    # except using our own grayscale reconstruction
    resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
    shifted_img = image - h - resolution
    reconstructed = reconstruct_fast_hybrid_python(shifted_img, image, radius)
    residue_img = image - reconstructed
    return (residue_img >= h).astype(np.uint8)


def cython_h_maxima(image, h=0.075, radius=2):
    # This is mostly copied from scikit h_maxima
    # except using our own grayscale reconstruction
    resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
    shifted_img = image - h - resolution
    reconstructed = fast_hybrid_reconstruct(shifted_img, image, radius)
    residue_img = image - reconstructed
    return (residue_img >= h).astype(np.uint8)
