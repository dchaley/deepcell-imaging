import numpy as np
from fasthybridreconstruct import fast_hybrid_reconstruct


def cython_reconstruct_wrapper(marker, mask, footprint=None):
    mask = np.copy(mask)

    if footprint is None:
        footprint = np.ones((3, 3), dtype=bool)
    else:
        footprint = footprint.astype(bool, copy=True)

    fast_hybrid_reconstruct(marker, mask, footprint)
    return mask
