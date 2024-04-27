import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import random
import skimage.morphology.grayreconstruct
import time
import timeit

from fast_reconstruct_wrapper import cython_reconstruct_wrapper


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
# depends on: https://github.com/dchaley/deepcell-imaging/issues/118
# @pytest.mark.parametrize(
#     "dimensions",
#     [
#         1,
#         2,
#         3,
#         4,
#     ],
# )
@pytest.mark.parametrize(
    "rows",
    [
        10,
        100,
        1000,
    ],
)
@pytest.mark.parametrize(
    "cols",
    [
        10,
        100,
        1000,
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        "dilation",
        "erosion",
    ],
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
def test_random_data(dtype, rows, cols, method, random_seed):
    """Test reconstruction on a random 100x100 image."""

    random.seed(random_seed)

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

    t = timeit.default_timer()
    cython_result = cython_reconstruct_wrapper(image, mask, method=method)
    print("Cython time:", timeit.default_timer() - t)
    t = timeit.default_timer()
    scikit_result = skimage.morphology.grayreconstruct.reconstruction(
        image, mask, method=method
    )
    print("Scikit time:", timeit.default_timer() - t)

    assert_array_almost_equal(cython_result, scikit_result)
