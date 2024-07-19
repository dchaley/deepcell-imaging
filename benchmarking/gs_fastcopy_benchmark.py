# This file makes assumptions about where data files are located.

from timeit import default_timer

import gs_fastcopy
import numpy as np
import smart_open

BUCKET = "davids-genomics-data-public"
BENCHMARKING_ROOT = "gs://{bucket}/benchmarking/2024-07-18-gs-fastcopy".format(
    bucket=BUCKET
)

DOWNLOAD_FILE_NPZ = "{root}/arrays.npz".format(root=BENCHMARKING_ROOT)
DOWNLOAD_FILE_NPZ_GZ = "{root}/arrays.npz.gz".format(root=BENCHMARKING_ROOT)

UPLOAD_FILE_NPZ = "{root}/tmp.npz".format(root=BENCHMARKING_ROOT)
UPLOAD_FILE_NPZ_GZ = "{root}/tmp.npz.gz".format(root=BENCHMARKING_ROOT)

LOCAL_FILE_NPZ = "raw_predictions.npz"


def load_numpy_arrays() -> dict:
    with np.load(LOCAL_FILE_NPZ) as npz_contents:
        return dict(npz_contents)


def get_numpy_array_names() -> list:
    with np.load(LOCAL_FILE_NPZ) as npz_contents:
        return list(npz_contents.keys())


# Uploads in serial.
def smart_open_upload_uncompressed(numpy_arrays: dict):
    with smart_open.open(UPLOAD_FILE_NPZ, "wb") as f:
        np.savez(f, **numpy_arrays)


# Downloads in serial.
def smart_open_download_uncompressed(array_names: list):
    with smart_open.open(DOWNLOAD_FILE_NPZ, "rb") as f:
        with np.load(f) as npz_contents:
            for name in array_names:
                _arr = npz_contents[name]


# Uploads in serial, via np.savez_compressed.
def smart_open_upload_savez_compressed(numpy_arrays: dict):
    with smart_open.open(UPLOAD_FILE_NPZ, "wb") as f:
        np.savez_compressed(f, **numpy_arrays)


# Downloads in serial & decompresses in-flight.
def smart_open_download_compressed(array_names: list):
    with smart_open.open(DOWNLOAD_FILE_NPZ_GZ, "rb") as f:
        with np.load(f) as npz_contents:
            for name in array_names:
                _arr = npz_contents[name]


# Uploads in parallel.
def gs_fastcopy_upload_uncompressed(numpy_arrays: dict):
    with gs_fastcopy.write(UPLOAD_FILE_NPZ) as f:
        np.savez(f, **numpy_arrays)


# Downloads in parallel.
def gs_fastcopy_download_uncompressed(array_names: list):
    with gs_fastcopy.read(DOWNLOAD_FILE_NPZ) as f:
        with np.load(f) as npz_contents:
            for name in array_names:
                _arr = npz_contents[name]


# Uploads in parallel, but gzips before.
def gs_fastcopy_upload_compressed(numpy_arrays: dict):
    with gs_fastcopy.write(UPLOAD_FILE_NPZ_GZ) as f:
        np.savez(f, **numpy_arrays)


# Downloads in parallel then decompresses..
def gs_fastcopy_download_compressed(array_names: list):
    with gs_fastcopy.read(DOWNLOAD_FILE_NPZ_GZ) as f:
        with np.load(f) as npz_contents:
            for name in array_names:
                _arr = npz_contents[name]


def main():
    arrays = load_numpy_arrays()
    array_names = get_numpy_array_names()

    print("test,time")

    # Run each benchmark.

    t = default_timer()
    smart_open_upload_uncompressed(arrays)
    print("smart_open_upload_uncompressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    smart_open_download_uncompressed(array_names)
    print("smart_open_download_uncompressed,%s" % round(default_timer() - t, 2))

    # It looks like the numpy savez implementation seeks which smart_open gzip doesn't support.
    # Disabling: we never actually used this (we used savez_compressed instead).
    # t = default_timer()
    # smart_open_upload_compressed(arrays)
    # print("smart_open_upload_compressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    smart_open_upload_savez_compressed(arrays)
    print("smart_open_upload_savez_compressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    smart_open_download_compressed(array_names)
    print("smart_open_download_compressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    gs_fastcopy_upload_uncompressed(arrays)
    print("gs_fastcopy_upload_uncompressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    gs_fastcopy_download_uncompressed(array_names)
    print("gs_fastcopy_download_uncompressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    gs_fastcopy_upload_compressed(arrays)
    print("gs_fastcopy_upload_compressed,%s" % round(default_timer() - t, 2))

    t = default_timer()
    gs_fastcopy_download_compressed(array_names)
    print("gs_fastcopy_download_compressed,%s" % round(default_timer() - t, 2))


if __name__ == "__main__":
    main()
