import logging

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import random
import skimage.morphology.grayreconstruct
import time

from benchmark_utils import opencv_reconstruct
from fast_reconstruct_wrapper import cython_reconstruct_wrapper

xfail = pytest.mark.xfail


@pytest.mark.parametrize(
    "dtype",
    [
        # OpenCV only supports: CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
        np.uint8,
        # np.uint16, # commented for dev
        np.int16,
        # np.float32, # commented for dev
        np.float64,
    ],
)
# depends on: https://github.com/dchaley/deepcell-imaging/issues/118
# @pytest.mark.parametrize(
#     "dimensions",
#       [1, 2, 3, 4],
#     ],
# )
@pytest.mark.parametrize(
    "rows",
    [11, 96, 1023],
)
@pytest.mark.parametrize(
    "cols",
    [15, 130, 999],
)
@pytest.mark.parametrize(
    "method",
    ["dilation"],
)
@pytest.mark.parametrize(
    "footprint_rows",
    [3, 5, 7, 9],
)
@pytest.mark.parametrize(
    "footprint_cols",
    [3, 5, 7, 9],
)
@pytest.mark.parametrize(
    "random_seed",
    [
        123,
        54321,
        422442,
        # While non-deterministic, this should still *never* fail
        int(time.time()),
    ],
)
def test_random_data(
    dtype, rows, cols, method, footprint_rows, footprint_cols, random_seed
):
    """Test reconstruction on a random 100x100 image."""

    np.random.seed(random_seed)

    shape = (rows, cols)

    if issubclass(dtype, np.floating):
        dtype_min, dtype_max = np.iinfo(np.int8).min, np.iinfo(np.int8).max
        image = (
            np.random.random_sample(size=shape) * (dtype_max - dtype_min) + dtype_min
        ).astype(dtype)
        mask = (
            np.random.random_sample(size=shape) * (dtype_max - dtype_min) + dtype_min
        ).astype(dtype)
    else:
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        image = np.random.randint(dtype_min, dtype_max, size=shape, dtype=dtype)
        mask = np.random.randint(dtype_min, dtype_max, size=shape, dtype=dtype)

    # The image & mask must follow these constraints.
    if method == "dilation":
        mask = np.maximum(image, mask, out=mask)
    else:
        mask = np.minimum(image, mask, out=mask)

    # To finish https://github.com/dchaley/deepcell-imaging/issues/118,
    # make this n-dimensional not 2d.
    footprint = np.random.randint(
        0, 2, size=(footprint_rows, footprint_cols), dtype=np.uint8
    )
    # Center point has to be true– the neighborhood always
    # includes the center point.
    footprint[footprint_rows // 2, footprint_cols // 2] = 1

    print("opencv")

    opencv_result = opencv_reconstruct(
        image.copy(),
        mask.copy(),
        footprint.copy(),
        # (0, 0),
    )

    print("cython")

    cython_result = cython_reconstruct_wrapper(
        image.copy(), mask.copy(), method=method, footprint=footprint.copy()
    )

    assert_array_almost_equal(cython_result, opencv_result)
