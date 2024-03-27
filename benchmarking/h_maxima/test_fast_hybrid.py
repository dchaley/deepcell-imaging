import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from fasthybridreconstruct import point_to_linear_python, linear_to_point_python


def test_point_to_linear():
    point = np.array([1, 2, 3])
    shape = np.array([4, 5, 6])
    linear = point_to_linear_python(point, shape, len(shape))
    assert linear == 3 + 2 * 6 + 1 * 6 * 5


def test_linear_to_point():
    linear = 3 + 2 * 6 + 1 * 6 * 5
    shape = np.array([4, 5, 6])
    point = linear_to_point_python(linear, shape, len(shape))
    assert_array_almost_equal(point, [1, 2, 3])
