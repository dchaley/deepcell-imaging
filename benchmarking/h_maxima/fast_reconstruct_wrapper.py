import numpy as np
from fasthybridreconstruct import (
    fast_hybrid_reconstruct,
    METHOD_DILATION,
    METHOD_EROSION,
)
from skimage._shared.utils import _supported_float_type


def cython_reconstruct_wrapper(
    marker, mask, footprint=None, inplace=False, method="dilation"
):
    if method == "dilation":
        method = METHOD_DILATION
    elif method == "erosion":
        method = METHOD_EROSION
    else:
        raise ValueError(
            "Reconstruction method can be 'dilation' or 'erosion', not '%s'." % method
        )

    if method == METHOD_DILATION and np.any(marker > mask):
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )
    elif method == METHOD_EROSION and np.any(marker < mask):
        raise ValueError(
            "Intensity of seed image must be greater than that "
            "of the mask image for reconstruction by erosion."
        )

    if inplace:
        marker = marker
    else:
        marker = marker.astype(_supported_float_type(mask.dtype), copy=True)

    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)
    else:
        footprint = footprint.astype(bool, copy=True)

    fast_hybrid_reconstruct(marker, mask, footprint, method)
    return marker
