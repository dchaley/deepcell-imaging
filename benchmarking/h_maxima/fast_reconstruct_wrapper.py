import numpy as np
from fasthybridreconstruct import fast_hybrid_reconstruct
from skimage._shared.utils import _supported_float_type


def cython_reconstruct_wrapper(marker, mask, footprint=None, inplace=False):
    if inplace:
        marker = marker
    else:
        marker = marker.astype(_supported_float_type(mask.dtype), copy=True)

    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)
    else:
        footprint = footprint.astype(bool, copy=True)

    fast_hybrid_reconstruct(marker, mask, footprint)
    return marker
