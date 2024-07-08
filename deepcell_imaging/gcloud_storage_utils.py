from contextlib import contextmanager
import os
import subprocess
import tempfile


@contextmanager
def reader(gs_uri):
    """
    Context manager for reading a file from Google Cloud Storage.

    Usage:
    ```
    with reader('gs://my-bucket/my-file.npz') as f:
        npz = np.load(f)
    ```

    This will download the file to a temporary directory, and
    open it for reading. When the 'with' block exits, the file
    will be closed and the temporary directory will be deleted.
    """
    with tempfile.TemporaryDirectory() as tmp:
        if gs_uri.endswith(".gz"):
            path = os.path.join(tmp, "downloaded_file.gz")
        else:
            path = os.path.join(tmp, "downloaded_file")

        # Transfer the file.
        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", gs_uri, path])

        # If necessary, decompress the file before reading.
        # unpigz is a parallel gunzip implementation that's
        # much faster when hardware is available.
        if path.endswith(".gz"):
            subprocess.run(["unpigz", path])
            path = path[:-3]

        with open(path, "rb") as f:
            yield f


@contextmanager
def writer(gs_uri):
    # Create a temporary scratch directory.
    # Will be deleted when the 'with' closes.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We need an actual filename within the scratch directory.
        buffer_file_name = os.path.join(tmp_dir, "file_to_upload")

        # Yield the file object for the caller to write.
        with open(buffer_file_name, "wb") as tmp_file:
            yield tmp_file

        # If requested, compress the file before uploading.
        # pigz is a parallel gzip implementation that's
        # much faster than numpy's savez_compressed.
        if gs_uri.endswith(".gz"):
            # TODO: handle errors
            subprocess.run(["pigz", buffer_file_name])
            buffer_file_name += ".gz"

        # TODO: handle errors
        subprocess.run(["gcloud", "storage", "cp", buffer_file_name, gs_uri])
