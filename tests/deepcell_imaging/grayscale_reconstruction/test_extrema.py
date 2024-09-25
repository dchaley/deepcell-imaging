"""
This file was copied from:
https://github.com/scikit-image/scikit-image

Specifically:
https://github.com/scikit-image/scikit-image/blob/d15a5f7c8292cb19b84bf8628df28eaf46f60476/skimage/morphology/tests/test_extrema.py

Licensed under BSD-3
"""

import math

import numpy as np
from skimage._shared.testing import expected_warnings

from deepcell_imaging.image_processing import extrema

eps = 1e-12


def diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = ((a - b) ** 2).sum()
    return math.sqrt(t)


class TestExtrema:

    def test_h_maxima(self):
        """h-maxima for various data types"""

        data = np.array(
            [
                [10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
                [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
                [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
                [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
                [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
                [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
                [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
                [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
            ],
            dtype=np.uint8,
        )

        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
            data = data.astype(dtype)
            out = extrema.h_maxima(data, 40)

            error = diff(expected_result, out)
            assert error < eps

    def test_extrema_float(self):
        """specific tests for float type"""
        data = np.array(
            [
                [0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11],
                [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13],
                [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18, 0.60, 0.60, 0.15],
                [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19, 0.60, 0.60, 0.16],
                [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
                [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19, 0.19, 0.18, 0.16],
                [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
                [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19, 1.0, 1.0, 0.16],
                [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15],
                [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13],
            ],
            dtype=np.float32,
        )

        out = extrema.h_maxima(data, 0.003)
        expected_result = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        error = diff(expected_result, out)
        assert error < eps

    def test_h_maxima_float_image(self):
        """specific tests for h-maxima float image type"""
        w = 10
        x, y = np.mgrid[0:w, 0:w]
        data = 20 - 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
        data[2:4, 2:4] = 40
        data[2:4, 7:9] = 60
        data[7:9, 2:4] = 80
        data[7:9, 7:9] = 100
        data = data.astype(np.float32)

        expected_result = np.zeros_like(data)
        expected_result[(data > 19.9)] = 1.0

        for h in [1.0e-12, 1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1, 0.1]:
            out = extrema.h_maxima(data, h)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_maxima_float_h(self):
        """specific tests for h-maxima float h parameter"""
        data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0],
                [0, 3, 4, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        h_vals = np.linspace(1.0, 2.0, 100)
        failures = 0

        for h in h_vals:
            if h % 1 != 0:
                msgs = ["possible precision loss converting image"]
            else:
                msgs = []

            with expected_warnings(msgs):
                maxima = extrema.h_maxima(data, h)

            if maxima[2, 2] == 0:
                failures += 1

        assert failures == 0

    def test_h_maxima_large_h(self):
        """test that h-maxima works correctly for large h"""
        data = np.array(
            [
                [10, 10, 10, 10, 10],
                [10, 13, 13, 13, 10],
                [10, 13, 14, 13, 10],
                [10, 13, 13, 13, 10],
                [10, 10, 10, 10, 10],
            ],
            dtype=np.uint8,
        )

        maxima = extrema.h_maxima(data, 5)
        assert np.sum(maxima) == 0

        data = np.array(
            [
                [10, 10, 10, 10, 10],
                [10, 13, 13, 13, 10],
                [10, 13, 14, 13, 10],
                [10, 13, 13, 13, 10],
                [10, 10, 10, 10, 10],
            ],
            dtype=np.float32,
        )

        maxima = extrema.h_maxima(data, 5.0)
        assert np.sum(maxima) == 0
