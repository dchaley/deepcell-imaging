import numpy as np
import smart_open
import zipfile


# Adapted from: https://stackoverflow.com/questions/35990775/finding-shape-of-saved-numpy-array-npy-or-npz-without-loading-into-memory
# (Added smart_open wrapper for streaming remote access)
def npz_headers(npz):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(smart_open.open(npz, mode="rb")) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype
