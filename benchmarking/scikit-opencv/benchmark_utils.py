import cv2
import numpy as np
import sys
from skimage.morphology import h_maxima
from skimage.morphology import disk
import timeit


def scikit_h_maxima(image, h=0.075, radius=2):
    return h_maxima(image=image, h=h, footprint=disk(radius))


# Note that this modifies the marker image in place
def opencv_reconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 2):
    kernel = disk(radius)

    # .item() converts the numpy scalar to a python scalar
    pad_value = np.min(marker).item()

    # Create an output buffer
    expanded = np.ndarray.copy(marker)

    while True:
        expanded = cv2.dilate(
            src=marker,
            dst=expanded,
            kernel=kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=pad_value,
        )
        expanded = np.fmin(expanded, mask)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded

        np.copyto(dst=marker, src=expanded)


def opencv_h_maxima(image, h=0.075, radius=2):
    # This is mostly copied from scikit h_maxima
    # except using our own grayscale reconstruction
    resolution = 2 * np.finfo(image.dtype).resolution * np.abs(image)
    shifted_img = image - h - resolution
    reconstructed = opencv_reconstruct(shifted_img, image, radius)
    residue_img = image - reconstructed
    return (residue_img >= h).astype(np.uint8)


def main(marker_filename):
    # Load the data
    with np.load(marker_filename) as loader:
        marker_data = np.load(marker_filename)["arr_0"]

    # Run the benchmark
    t = timeit.default_timer()
    scikit_result = scikit_h_maxima(marker_data[0, ..., 0])
    print("scikit h_maxima: %s s" % (timeit.default_timer() - t))

    t = timeit.default_timer()
    opencv_result = opencv_h_maxima(marker_data[0, ..., 0])
    print("opencv h_maxima: %s s" % (timeit.default_timer() - t))

    diff = np.abs(scikit_result - opencv_result)
    print("results sizes: %s, %s" % (scikit_result.shape, opencv_result.shape))
    print("results     equal: %s" % (diff == 0).sum())
    print("results different: %s" % (diff != 0).sum())
    print("smallest difference: %s" % (diff.min()))
    print(" largest difference: %s" % (diff.max()))
    print(" average difference: %s" % (diff.mean()))


if __name__ == "__main__":
    main(sys.argv[1])
