
from contextlib import contextmanager
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


@contextmanager
def writer(gs_uri):
    # Create a temporary scratch directory.
    # Will be deleted when the 'with' closes.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We need an actual filename within the scratch directory.
        buffer_file_name = os.path.join(tmp_dir, 'file_to_upload')

        # Yield the file object for the caller to write.
        with open(buffer_file_name, 'wb') as tmp_file:
            yield tmp_file

        # If requested, compress the file before uploading.
        # pigz is a parallel gzip implementation that's
        # much faster than numpy's savez_compressed.
        if gs_uri.endswith('.gz'):
            # TODO: handle errors
            subprocess.run(["pigz", buffer_file_name])
            buffer_file_name += '.gz'

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", buffer_file_name, gs_uri])
