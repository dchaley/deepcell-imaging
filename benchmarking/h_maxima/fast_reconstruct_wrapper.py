import numpy as np
from fasthybridreconstruct import (
    fast_hybrid_reconstruct,
    METHOD_DILATION,
    METHOD_EROSION,
)
from skimage._shared.utils import _supported_float_type


def cython_reconstruct_wrapper(
    image, mask, method="dilation", footprint=None, offset=None, inplace=False
):
    if method == "dilation":
        method = METHOD_DILATION
    elif method == "erosion":
        method = METHOD_EROSION
    else:
        raise ValueError(
            "Reconstruction method can be 'dilation' or 'erosion', not '%s'." % method
        )

    if method == METHOD_DILATION and np.any(image > mask):
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )
    elif method == METHOD_EROSION and np.any(image < mask):
        raise ValueError(
            "Intensity of seed image must be greater than that "
            "of the mask image for reconstruction by erosion."
        )

    if footprint is None:
        footprint = np.ones([3] * image.ndim, dtype=np.uint8)
    else:
        footprint = footprint.astype(np.uint8, copy=True)

    if offset is None:
        if not all([d % 2 == 1 for d in footprint.shape]):
            raise ValueError("Footprint dimensions must all be odd")
        offset = np.array([d // 2 for d in footprint.shape])
    else:
        if offset.shape[0] != footprint.ndim:
            raise ValueError("Offset length and footprint ndims must be equal.")
        if not all([(0 <= o < d) for o, d in zip(offset, footprint.shape)]):
            raise ValueError("Offset must be included inside footprint")

    if inplace:
        image = image
    else:
        image = image.astype(_supported_float_type(mask.dtype), copy=True)

    # The existing skimage code is inconsistent in how it creates the offset.
    # See the offset is None block: it creates offsets of different shapes.
    # The default offset only has 1 dimension (a list of coordinates), but
    # a provided offset must have the same dimensions as the footprint,
    # shaped as a single element in each dimension.
    # Somehow, it all still works.🤔
    #
    # But I couldn't figure out how to pass a C reference to the data.
    # This creates a new bytes buffer, using the contiguous array data.
    # Cython knows how to wire this into a pointer for the C function.
    # When we support n-dimensions and need to figure out pointers for
    # all types, remove this…
    offset = offset.astype(np.uint8, copy=True)

    fast_hybrid_reconstruct(
        image=image,
        mask=mask,
        footprint=footprint,
        method=method,
        offset=offset,
    )
    return image
