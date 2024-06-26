
import io
import numpy as np
import os
import subprocess
import tempfile


def fetch_file(gs_uri):
    """
    Fetch a file from Google Cloud Storage using its URI.

    Returns it as a BytesIO object.

    Under the hood, calls out to the `gcloud storage` command-line tool,
    which has optimized performance for large data transfers (for example,
    parallel/chunked transfers).
    """
    with tempfile.TemporaryDirectory() as tmp:
        if gs_uri.endswith('.gz'):
            path = os.path.join(tmp, 'downloaded_file.gz')
        else:
            path = os.path.join(tmp, 'downloaded_file')

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", gs_uri, path])

        if path.endswith('.gz'):
            subprocess.run(["unpigz", path])
            path = path[:-3]

        with open(path, 'rb') as f:
            return io.BytesIO(f.read())


def write_npz_file(gs_uri, **named_arrays):
    """
    Write a BytesIO object to Google Cloud Storage using its URI.

    Under the hood, calls out to the `gcloud storage` command-line tool,
    which has optimized performance for large data transfers (for example,
    parallel/chunked transfers).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'file_to_upload.npz')
        np.savez(path, **named_arrays)

        if gs_uri.endswith(".gz"):
            subprocess.run(["pigz", path])
            path = path + ".gz"

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", path, gs_uri])
