
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
        path = os.path.join(tmp, 'downloaded_file')

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", gs_uri, path])

        with open(path, 'rb') as f:
            return io.BytesIO(f.read())


def write_npz_file(gs_uri, **named_arrays):
    """
    Write a BytesIO object to Google Cloud Storage using its URI.

    Under the hood, calls out to the `gcloud storage` command-line tool,
    which has optimized performance for large data transfers (for example,
    parallel/chunked transfers).
    """
    with tempfile.NamedTemporaryFile() as tmp:
        np.savez_compressed(tmp, **named_arrays)
        tmp.flush()

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", tmp.name, gs_uri])
