# This file was lifted wholesale from scikit-image
# https://github.com/scikit-image/scikit-image/blob/0e35d8040d0b0bb409de41f3c0f024f941e09d8a/skimage/morphology/tests/test_reconstruction.py
# and modified to use my own reconstruction function

import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from skimage._shared.utils import _supported_float_type

from grayscale_reconstruction.fast_hybrid import (
    fast_hybrid_reconstruct as reconstruction,
)

xfail = pytest.mark.xfail


def test_zeros():
    """Test reconstruction with image and mask of zeros"""

    # With & without explicit offset
    assert_array_almost_equal(reconstruction(np.zeros((5, 7)), np.zeros((5, 7))), 0)
    assert_array_almost_equal(
        reconstruction(np.zeros((5, 7)), np.zeros((5, 7)), offset=np.array([1, 1])),
        0,
    )


def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert_array_almost_equal(reconstruction(np.ones((7, 5)), np.ones((7, 5))), 1)


def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = np.ones((5, 5))
    mask = np.ones((5, 5)) * 2
    assert_array_almost_equal(reconstruction(image, mask), 1)


def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = np.ones((5, 5))
    image[2, 2] = 2
    mask = np.ones((5, 5)) * 3
    assert_array_almost_equal(reconstruction(image, mask), 2)


# minsize chosen to test sizes covering use of 8, 16 and 32-bit integers
# internally
@pytest.mark.parametrize("minsize", [None, 200, 20000, 40000, 80000])
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.float32,
    ],
)
def test_two_image_peaks(minsize, dtype):
    """Test reconstruction with two peak pixels isolated by the mask"""
    image = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 3, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=dtype,
    )

    mask = np.array(
        [
            [4, 4, 4, 1, 1, 1, 1, 1, 1],
            [4, 4, 4, 1, 1, 1, 1, 1, 1],
            [4, 4, 4, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 4, 4, 4, 1],
            [1, 1, 1, 1, 1, 4, 4, 4, 1],
            [1, 1, 1, 1, 1, 4, 4, 4, 1],
        ],
        dtype=dtype,
    )

    expected = np.array(
        [
            [2, 2, 2, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 3, 3, 3, 1],
            [1, 1, 1, 1, 1, 3, 3, 3, 1],
            [1, 1, 1, 1, 1, 3, 3, 3, 1],
        ],
        dtype=dtype,
    )
    if minsize is not None:
        # increase data size by tiling (done to test various int types)
        nrow = math.ceil(math.sqrt(minsize / image.size))
        ncol = math.ceil(minsize / (image.size * nrow))
        image = np.tile(image, (nrow, ncol))
        mask = np.tile(mask, (nrow, ncol))
        expected = np.tile(expected, (nrow, ncol))
    out = reconstruction(image, mask)
    assert out.dtype == _supported_float_type(mask.dtype)
    assert_array_almost_equal(out, expected)


def test_zero_image_one_mask():
    """Test reconstruction with an image of all zeros and a mask that's not"""
    result = reconstruction(np.zeros((10, 10)), np.ones((10, 10)))
    assert_array_almost_equal(result, 0)


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
        # np.float16,
        np.float32,
        np.float64,
    ],
)
def test_fill_hole(dtype):
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = np.array([[0, 8, 8, 8, 8, 8, 8, 8, 8, 0]], dtype=dtype)
    mask = np.array([[0, 3, 6, 2, 1, 1, 1, 4, 2, 0]], dtype=dtype)
    result = reconstruction(seed, mask, method="erosion")
    assert result.dtype == _supported_float_type(mask.dtype)
    expected = np.array([[0, 3, 6, 4, 4, 4, 4, 4, 2, 0]], dtype=dtype)
    assert_array_almost_equal(result, expected)


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
        # np.float16,
        np.float32,
        np.float64,
    ],
)
def test_fill_hole_1d(dtype):
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = np.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0], dtype=dtype)
    mask = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0], dtype=dtype)
    result = reconstruction(seed, mask, method="erosion")
    assert result.dtype == _supported_float_type(mask.dtype)
    expected = np.array([0, 3, 6, 4, 4, 4, 4, 4, 2, 0], dtype=dtype)
    assert_array_almost_equal(result, expected)


def test_invalid_seed():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed * 2, mask, method="dilation")
    with pytest.raises(ValueError):
        reconstruction(seed * 0.5, mask, method="erosion")


def test_invalid_footprint():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((4, 4)))
    with pytest.raises(ValueError):
        reconstruction(seed, mask, footprint=np.ones((3, 4)))
    reconstruction(seed, mask, footprint=np.ones((3, 3)))


def test_invalid_method():
    seed = np.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    with pytest.raises(ValueError):
        reconstruction(seed, mask, method="foo")


def test_invalid_offset_not_none():
    """Test reconstruction with invalid not None offset parameter"""
    image = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    mask = np.array(
        [
            [4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 1, 1, 1, 1, 1],
            [4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 4, 4, 4],
            [1, 1, 1, 1, 1, 4, 4, 4],
            [1, 1, 1, 1, 1, 4, 4, 4],
        ]
    )
    with pytest.raises(ValueError):
        reconstruction(
            image,
            mask,
            footprint=np.ones((3, 3)),
            offset=np.array([3, 0]),
        )


def test_offset_not_none_1d():
    """Test reconstruction with valid offset parameter"""
    seed = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = np.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    expected = np.array([0, 6, 6, 4, 4, 4, 4, 4, 2, 0])

    assert_array_almost_equal(
        reconstruction(
            seed, mask, method="dilation", footprint=np.ones(3), offset=np.array([0])
        ),
        expected,
    )


def test_offset_not_none():
    """Test reconstruction with valid offset parameter"""
    seed = np.array([[0, 3, 6, 2, 1, 1, 1, 4, 2, 0]], dtype=np.int16)
    mask = np.array([[0, 8, 6, 8, 8, 8, 8, 4, 4, 0]], dtype=np.int16)
    expected = np.array([[0, 6, 6, 4, 4, 4, 4, 4, 2, 0]], dtype=np.int16)

    assert_array_almost_equal(
        reconstruction(
            seed,
            mask,
            method="dilation",
            footprint=np.array([[1, 1, 1]], np.uint8),
            offset=np.array([0, 0]),
        ),
        expected,
    )


def test_arbitrary_2x2():
    seed = np.array(
        [
            [49, -1],
            [-18, 56],
        ],
        dtype=np.int16,
    )
    mask = np.array(
        [
            [49, -1],
            [37, 57],
        ],
        dtype=np.int16,
    )
    footprint = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    # Proof of work following iterations of parallel method:
    # (repeat point-wise maximum filtered min-wise with mask, until convergence)
    #   0, 0: min(max( 49,  56), 49) =  49
    #   0, 1: min(max( -1, -18), -1) =  -1
    #   1, 0: min(max(-18)     , 37) = -18
    #   1, 1: min(max( 56,  49), 57) =  56
    # Immediate convergence.
    expected = np.array(
        [
            [49, -1],
            [-18, 56],
        ],
        dtype=np.int16,
    )

    fast_hybrid_result = reconstruction(
        np.ndarray.copy(seed),
        np.ndarray.copy(mask),
        method="dilation",
        footprint=np.ndarray.copy(footprint),
    )

    assert_array_almost_equal(
        fast_hybrid_result,
        expected,
    )


def test_arbitrary_3x3():
    seed = np.array(
        [
            [80, 108, 57],
            [225, 86, 50],
            [102, 148, 126],
        ],
        dtype=np.int16,
    )
    mask = np.array(
        [
            [111, 108, 57],
            [225, 218, 50],
            [104, 162, 189],
        ],
        dtype=np.int16,
    )
    footprint = np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    # Proof of work following iterations of parallel method:
    # (repeat point-wise maximum filtered min-wise with mask, until convergence)
    # https://docs.google.com/spreadsheets/d/15TntEi694OUy3ZjAdAdY2Ax6FTdCsCXiqGfpxpV7-iY/edit#gid=1682049169
    expected = np.array(
        [
            [111, 108, 57],
            [225, 162, 50],
            [104, 162, 162],
        ],
        dtype=np.int16,
    )

    fast_hybrid_result = reconstruction(
        np.ndarray.copy(seed),
        np.ndarray.copy(mask),
        method="dilation",
        footprint=np.ndarray.copy(footprint),
    )

    assert_array_almost_equal(
        fast_hybrid_result,
        expected,
    )


def test_arbitrary_5x5():
    seed = np.array(
        [
            [184, 83, 30, 47, 184],
            [180, 80, 108, 57, 230],
            [251, 225, 86, 50, 105],
            [216, 102, 148, 126, 90],
            [35, 73, 208, 97, 100],
        ],
        dtype=np.int16,
    )
    mask = np.array(
        [
            [184, 195, 177, 180, 184],
            [180, 111, 108, 57, 230],
            [251, 225, 218, 50, 136],
            [216, 104, 162, 189, 90],
            [217, 135, 208, 249, 125],
        ],
        dtype=np.int16,
    )
    footprint = np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )
    # Proof of work following iterations of parallel method:
    # (repeat point-wise maximum filtered min-wise with mask, until convergence)
    # https://docs.google.com/spreadsheets/d/15TntEi694OUy3ZjAdAdY2Ax6FTdCsCXiqGfpxpV7-iY/edit#gid=1552646482
    expected = np.array(
        [
            [184, 180, 111, 108, 184],
            [180, 111, 108, 57, 230],
            [251, 225, 162, 50, 136],
            [216, 104, 162, 189, 90],
            [216, 135, 208, 189, 125],
        ],
        dtype=np.int16,
    )

    fast_hybrid_result = reconstruction(
        np.ndarray.copy(seed),
        np.ndarray.copy(mask),
        method="dilation",
        footprint=np.ndarray.copy(footprint),
    )

    assert_array_almost_equal(
        fast_hybrid_result,
        expected,
    )


def test_arbitrary_3x3_offset():
    seed = np.array(
        [
            [19966, 12875, -17043],
            [13956, -4738, -14016],
            [-15038, -4688, 28636],
        ],
        dtype=np.int16,
    )
    mask = np.array(
        [
            [21602, 12875, 30438],
            [13956, -4738, 14362],
            [-15038, 14383, 30826],
        ],
        dtype=np.int16,
    )
    footprint = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    # Proof of work following iterations of parallel method:
    # (repeat point-wise maximum filtered min-wise with mask, until convergence)
    # https://docs.google.com/spreadsheets/d/15TntEi694OUy3ZjAdAdY2Ax6FTdCsCXiqGfpxpV7-iY/edit#gid=1863474562
    expected = np.array(
        [
            [19966, 12875, 19966],
            [13956, -4738, 13956],
            [-15038, 13956, 28636],
        ],
        dtype=np.int16,
    )

    offset = np.array([2, 2])

    fast_hybrid_result = reconstruction(
        seed.copy(),
        mask.copy(),
        method="dilation",
        footprint=footprint.copy(),
        offset=offset,
    )

    assert_array_almost_equal(
        fast_hybrid_result,
        expected,
    )


def test_arbitrary_3x3_offset_5x3_footprint():
    seed = np.array(
        [
            [254, 205, 75],
            [178, 109, 61],
            [132, 182, 126],
        ],
        dtype=np.uint8,
    )
    mask = np.array(
        [
            [254, 205, 176],
            [178, 220, 239],
            [132, 182, 126],
        ],
        dtype=np.uint8,
    )
    footprint = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    # Proof of work following iterations of parallel method:
    # (repeat point-wise maximum filtered min-wise with mask, until convergence)
    # https://docs.google.com/spreadsheets/d/15TntEi694OUy3ZjAdAdY2Ax6FTdCsCXiqGfpxpV7-iY/edit#gid=1552794151
    expected = np.array(
        [
            [254, 205, 176],
            [178, 132, 182],
            [132, 182, 126],
        ],
        dtype=np.uint8,
    )

    offset = np.array([0, 2])

    fast_hybrid_result = reconstruction(
        np.ndarray.copy(seed),
        np.ndarray.copy(mask),
        method="dilation",
        footprint=np.ndarray.copy(footprint),
        offset=offset,
    )

    assert_array_almost_equal(
        fast_hybrid_result,
        expected,
    )
