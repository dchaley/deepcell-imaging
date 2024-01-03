import numpy as np
from fasthybridreconstruct import fast_hybrid_reconstruct

def cython_reconstruct_wrapper(marker, mask):
    mask = np.copy(mask)
    fast_hybrid_reconstruct(marker, mask, 2)
    return mask

